"""THIS MODULE IS NOT TESTED YET.

IT IS WORK IN PROGRESS.

"""

import copy
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from dcegm.egm.aggregate_marginal_utility import (
    calculate_choice_probs_and_unsqueezed_logsum,
)
from dcegm.interface import get_state_choice_index_per_discrete_state
from dcegm.interpolation.interp1d import interp_value_on_wealth
from dcegm.interpolation.interp2d import interp2d_value_on_wealth_and_regular_grid
from dcegm.solve import get_solve_func_for_model


def create_individual_likelihood_function_for_model(
    model: Dict[str, Any],
    observed_states: Dict[str, int],
    observed_choices: np.array,
    params_all,
    unobserved_state_specs=None,
    return_model_solution=False,
    use_probability_of_observed_states=True,
):

    solve_func = get_solve_func_for_model(
        model=model,
    )
    if unobserved_state_specs is None:
        choice_prob_func = create_partial_choice_prob_calculation(
            observed_states=observed_states,
            observed_choices=observed_choices,
            model=model,
        )
    else:

        choice_prob_func = create_choice_prob_func_unobserved_states(
            model=model,
            observed_states=observed_states,
            observed_choices=observed_choices,
            unobserved_state_specs=unobserved_state_specs,
            use_probability_of_observed_states=use_probability_of_observed_states,
        )

    def individual_likelihood(params):
        params_update = params_all.copy()
        params_update.update(params)
        (
            value_solved,
            policy_solved,
            endog_grid_solved,
        ) = solve_func(params_update)

        choice_probs = choice_prob_func(
            value_in=value_solved,
            endog_grid_in=endog_grid_solved,
            params_in=params_update,
        )
        # Negative ll contributions are positive numbers. The smaller the better the fit
        # Add high fixed punishment for not explained choices
        neg_likelihood_contributions = (-jnp.log(choice_probs)).clip(max=999)

        if return_model_solution:
            solution = {
                "value": value_solved,
                "policy": policy_solved,
                "endog_grid": endog_grid_solved,
            }
            return neg_likelihood_contributions, solution
        else:
            return neg_likelihood_contributions

    return jax.jit(individual_likelihood)


def create_choice_prob_func_unobserved_states(
    model: Dict[str, Any],
    observed_states: Dict[str, int],
    observed_choices: np.array,
    unobserved_state_specs,
    use_probability_of_observed_states=True,
):
    # First prepare full observed states, choices and pre period states for weighting
    state_space_names = model["model_structure"]["discrete_states_names"] + ["wealth"]
    if len(model["options"]["exog_grids"]) == 2:
        second_cont_state_name = model["options"]["second_continuous_state_name"]
        state_space_names += [second_cont_state_name]

    unobserved_state_names = unobserved_state_specs["observed_bools_states"].keys()
    observed_bools = unobserved_state_specs["observed_bools_states"]

    # Create weighting vars by extracting states and choices
    weighting_vars = unobserved_state_specs["state_choices_weighing"]["states"]
    weighting_vars["choice"] = unobserved_state_specs["state_choices_weighing"][
        "choices"
    ]

    # Add unobserved states with appendix new and bools indicating if state is observed
    for state_name in unobserved_state_names:
        weighting_vars[state_name + "_new"] = observed_states[state_name]

    # Read out possible values for unobserved states
    unobserved_state_values = {}
    for state_name in unobserved_state_specs["observed_bools_states"].keys():
        if state_name in model["model_structure"]["exog_states_names"]:
            state_values = model["options"]["state_space"]["exogenous_processes"][
                state_name
            ]["states"]
        else:
            state_values = model["options"]["state_space"]["endogenous_states"][
                state_name
            ]
        unobserved_state_values[state_name] = state_values

    # Now create a list which contains dictionaries with ach dictionary
    # containing a unique combination of unobserved states. Note that this is
    # only tested for one state with two values.
    possible_states = [observed_states]
    weighting_vars_for_possible_states = [weighting_vars]
    for state_name in unobserved_state_names:
        # Create bool indicating if state is unobserved
        unobserved_state_bool = ~observed_bools[state_name]

        new_possible_states = []
        new_weighting_vars_for_possible_states = []
        for state_value in unobserved_state_values[state_name]:
            for possible_state in possible_states:
                possible_state[state_name][unobserved_state_bool] = state_value
                new_possible_states.append(copy.deepcopy(possible_state))
            # Same for pre period states
            for weighting_vars in weighting_vars_for_possible_states:
                weighting_vars[state_name + "_new"][unobserved_state_bool] = state_value
                new_weighting_vars_for_possible_states.append(
                    copy.deepcopy(weighting_vars)
                )
        # Now overwrite existing lists
        possible_states = new_possible_states
        weighting_vars_for_possible_states = new_weighting_vars_for_possible_states

    # Generate container for additional reweighting observed variables
    # As for these the weight function is called 2 times, we need to half the weight later
    observed_weights = np.ones(len(observed_choices), dtype=float)
    # For each observed variable, we divide by the number of possible values, as the weight function
    # is called for each observed value that often
    for state_name in unobserved_state_names:
        n_state_values = len(unobserved_state_values[state_name])

        observed_weights[observed_bools[state_name]] /= n_state_values

    # Create a list of partial choice probability functions for each unique
    # combination of unobserved states.
    partial_choice_probs_unobserved_states = []
    for states in possible_states:
        partial_choice_probs_unobserved_states.append(
            create_partial_choice_prob_calculation(
                observed_states=states,
                observed_choices=observed_choices,
                model=model,
            )
        )
    partial_weight_func = (
        lambda params_in, weight_vars: calculate_weights_for_each_state(
            params=params_in,
            weight_vars=weight_vars,
            options=model["options"]["model_params"],
            weight_func=unobserved_state_specs["weight_func"],
        )
    )

    n_obs = len(observed_choices)

    def choice_prob_func(value_in, endog_grid_in, params_in):
        choice_probs_final = jnp.zeros(n_obs, dtype=jnp.float64)
        integrate_out_weights = jnp.zeros(n_obs, dtype=jnp.float64)
        for partial_choice_prob, unobserved_state, weighting_vars in zip(
            partial_choice_probs_unobserved_states,
            possible_states,
            weighting_vars_for_possible_states,
        ):
            unobserved_weights = jax.vmap(
                partial_weight_func,
                in_axes=(None, 0),
            )(
                params_in,
                weighting_vars,
            )

            unweighted_choice_probs = partial_choice_prob(
                value_in=value_in,
                endog_grid_in=endog_grid_in,
                params_in=params_in,
            )

            weighted_choice_prob = jnp.nan_to_num(
                unweighted_choice_probs * unobserved_weights * observed_weights, nan=0.0
            )

            integrate_out_weights += unobserved_weights * observed_weights

            choice_probs_final += weighted_choice_prob

        if not use_probability_of_observed_states:
            choice_probs_final /= integrate_out_weights

        return choice_probs_final

    return choice_prob_func


def create_partial_choice_prob_calculation(
    observed_states,
    observed_choices,
    model,
):
    discrete_observed_state_choice_indexes = get_state_choice_index_per_discrete_state(
        states=observed_states,
        map_state_choice_to_index=model["model_structure"][
            "map_state_choice_to_index_with_proxy"
        ],
        discrete_states_names=model["model_structure"]["discrete_states_names"],
    )

    def partial_choice_prob_func(value_in, endog_grid_in, params_in):
        return calc_choice_prob_for_state_choices(
            value_solved=value_in,
            endog_grid_solved=endog_grid_in,
            params=params_in,
            states=observed_states,
            choices=observed_choices,
            state_choice_indexes=discrete_observed_state_choice_indexes,
            model=model,
        )

    return partial_choice_prob_func


def calc_choice_prob_for_state_choices(
    value_solved,
    endog_grid_solved,
    params,
    states,
    choices,
    state_choice_indexes,
    model,
):
    """This function interpolates the policy and value function for all agents.

    It uses the states at the beginning of period to select the solved policy and value
    and then interpolates the wealth at the beginning of period on them.

    """
    choice_prob_across_choices = calc_choice_probs_for_states(
        value_solved=value_solved,
        endog_grid_solved=endog_grid_solved,
        params=params,
        observed_states=states,
        state_choice_indexes=state_choice_indexes,
        model=model,
    )
    choice_probs = jnp.take_along_axis(
        choice_prob_across_choices, choices[:, None], axis=1
    )[:, 0]
    return choice_probs


def calc_choice_probs_for_states(
    value_solved,
    endog_grid_solved,
    params,
    observed_states,
    state_choice_indexes,
    model,
):
    value_grid_agent = jnp.take(
        value_solved, state_choice_indexes, axis=0, mode="fill", fill_value=jnp.nan
    )
    endog_grid_agent = jnp.take(endog_grid_solved, state_choice_indexes, axis=0)

    # Read out relevant model objects
    options = model["options"]
    choice_range = options["state_space"]["choices"]
    compute_utility = model["model_funcs"]["compute_utility"]

    if len(options["exog_grids"]) == 2:
        vectorized_interp2d = jax.vmap(
            jax.vmap(
                interp2d_value_for_state_in_each_choice,
                in_axes=(None, None, 0, 0, 0, None, None, None),
            ),
            in_axes=(0, 0, 0, 0, None, None, None, None),
        )
        # Extract second cont state name
        second_continuous_state_name = options["second_continuous_state_name"]
        second_cont_value = observed_states[second_continuous_state_name]

        value_per_agent_interp = vectorized_interp2d(
            observed_states,
            second_cont_value,
            endog_grid_agent,
            value_grid_agent,
            choice_range,
            params,
            options["exog_grids"]["second_continuous"],
            compute_utility,
        )

    else:
        vectorized_interp1d = jax.vmap(
            jax.vmap(
                interp1d_value_for_state_in_each_choice,
                in_axes=(None, 0, 0, 0, None, None),
            ),
            in_axes=(0, 0, 0, None, None, None),
        )

        value_per_agent_interp = vectorized_interp1d(
            observed_states,
            endog_grid_agent,
            value_grid_agent,
            choice_range,
            params,
            compute_utility,
        )

    model_funcs = model["model_funcs"]

    # The following allows to specify a function to return taste shock scales for each
    # state differently.
    if model_funcs["shock_functions"]["calc_taste_shock_scale_per_state"]:
        taste_shock_scale_per_state = model_funcs["shock_functions"][
            "taste_shock_scale_per_state"
        ](params=params, state_dict=observed_states)
        n_dims = len(value_per_agent_interp.shape)
        new_dims = (...,) + (None,) * (n_dims - 1)
        taste_shock_scale = taste_shock_scale_per_state[new_dims]

    else:
        taste_shock_scale = model_funcs["shock_functions"]["taste_shock_scale"](params)

    choice_prob_across_choices, _, _ = calculate_choice_probs_and_unsqueezed_logsum(
        choice_values_per_state=value_per_agent_interp,
        taste_shock_scale=taste_shock_scale,
    )
    return choice_prob_across_choices


def interp2d_value_for_state_in_each_choice(
    state,
    second_cont_state,
    endog_grid_agent,
    value_agent,
    choice,
    params,
    regular_grid,
    compute_utility,
):
    state_choice_vec = {**state, "choice": choice}

    value_interp = interp2d_value_on_wealth_and_regular_grid(
        regular_grid=regular_grid,
        wealth_grid=endog_grid_agent,
        value_grid=value_agent,
        regular_point_to_interp=second_cont_state,
        wealth_point_to_interp=state["wealth"],
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
    )

    return value_interp


def interp1d_value_for_state_in_each_choice(
    state,
    endog_grid_agent,
    value_agent,
    choice,
    params,
    compute_utility,
):
    state_choice_vec = {**state, "choice": choice}

    value_interp = interp_value_on_wealth(
        wealth=state["wealth"],
        endog_grid=endog_grid_agent,
        value=value_agent,
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
    )

    return value_interp


def calculate_weights_for_each_state(params, weight_vars, options, weight_func):
    """Calculate the weights for each state.

    Args:
        params (dict): Parameters.
        state_vec (dict): State vector.
        choice (int): Choice.
        options (dict): Options.
        weight_func (Callable): Weight function.

    Returns:
        float: Weight.

    """
    return weight_func(**weight_vars, params=params, options=options)
