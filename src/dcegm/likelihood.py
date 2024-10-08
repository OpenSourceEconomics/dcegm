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
            weight_full_states=True,
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
        return neg_likelihood_contributions

    return jax.jit(individual_likelihood)


def create_choice_prob_func_unobserved_states(
    model: Dict[str, Any],
    observed_states: Dict[str, int],
    observed_choices: np.array,
    unobserved_state_specs,
    weight_full_states=True,
):
    # First prepare full observed states, choices and pre period states for weighting
    full_mask = unobserved_state_specs["observed_bool"]
    if len(model["options"]["exog_grids"]) == 2:
        second_cont_state_name = model["options"]["second_continuous_state_name"]
        state_space_names = model["model_structure"]["discrete_states_names"] + [
            "wealth",
            second_cont_state_name,
        ]
    else:
        state_space_names = model["model_structure"]["discrete_states_names"] + [
            "wealth"
        ]

    full_observed_states = {
        name: observed_states[name][full_mask] for name in state_space_names
    }
    full_observed_choices = observed_choices[full_mask]
    # Now the states of last period for weighting and also the unobserved states
    # for this period
    pre_period_full_observed_states = {
        name: unobserved_state_specs["pre_period_states"][name][full_mask]
        for name in unobserved_state_specs["pre_period_states"].keys()
    }
    for state_name in unobserved_state_specs["states"]:
        pre_period_full_observed_states[state_name + "_new"] = full_observed_states[
            state_name
        ]

    # Finish with partial prob function for full observed states
    partial_choice_probs_full_observed_states = create_partial_choice_prob_calculation(
        observed_states=full_observed_states,
        observed_choices=full_observed_choices,
        model=model,
    )

    # Read out possible values for unobserved states
    unobserved_state_values = {}
    for state_name in unobserved_state_specs["states"]:
        if state_name in model["model_structure"]["exog_states_names"]:
            state_values = model["options"]["state_space"]["exogenous_processes"][
                state_name
            ]["states"]
        else:
            state_values = model["options"]["state_space"]["endogenous_states"][
                state_name
            ]
        unobserved_state_values[state_name] = state_values

    # Read out the observed states of the unobserved states
    unobserved_states = {
        name: observed_states[name][~full_mask] for name in state_space_names
    }
    # Also pre period states
    pre_period_unobserved_states = {
        name: unobserved_state_specs["pre_period_states"][name][~full_mask]
        for name in unobserved_state_specs["pre_period_states"].keys()
    }
    # Now add the new states which correspond to the states of this period
    for state_name in unobserved_state_specs["states"]:
        pre_period_unobserved_states[state_name + "_new"] = unobserved_states[
            state_name
        ]

    # Now create a list which contains dictionaries with ach dictionary
    # containing a unique combination of unobserved states. Note that this is
    # only tested for one state with two values.
    possible_unobserved_states = [unobserved_states]
    possible_pre_period_unobserved_states = [pre_period_unobserved_states]
    for state_name in unobserved_state_specs["states"]:
        new_possible_unobserved_states = []
        new_possible_pre_period_unobserved_states = []
        for state_value in unobserved_state_values[state_name]:
            for possible_state in possible_unobserved_states:
                possible_state[state_name][:] = state_value
                new_possible_unobserved_states.append(copy.deepcopy(possible_state))
            # Same for pre period states
            for pre_period_state in possible_pre_period_unobserved_states:
                pre_period_state[state_name + "_new"][:] = state_value
                new_possible_pre_period_unobserved_states.append(
                    copy.deepcopy(pre_period_state)
                )
        # Now overwrite existing lists
        possible_unobserved_states = new_possible_unobserved_states
        possible_pre_period_unobserved_states = (
            new_possible_pre_period_unobserved_states
        )

    # Create a list of partial choice probability functions for each unique
    # combination of unobserved states.
    partial_choice_probs_unobserved_states = []
    for unobserved_state in possible_unobserved_states:
        partial_choice_probs_unobserved_states.append(
            create_partial_choice_prob_calculation(
                observed_states=unobserved_state,
                observed_choices=observed_choices[~full_mask],
                model=model,
            )
        )
    partial_weight_func = (
        lambda params_in, states, choices: calculate_weights_for_each_state(
            params=params_in,
            state_vec=states,
            choice=choices,
            options=model["options"],
            weight_func=unobserved_state_specs["weight_func"],
        )
    )

    unobserved_states_index = jnp.where(~full_mask)[0]
    observed_states_index = jnp.where(full_mask)[0]

    def choice_prob_func(value_in, endog_grid_in, params_in):
        choice_probs_final = jnp.empty_like(observed_choices, dtype=jnp.float64)
        unobserved_probs = jnp.zeros_like(
            observed_choices[~full_mask], dtype=jnp.float64
        )
        objects = {}
        i = 0
        for partial_choice_prob, unobserved_state, pre_period_unobserved_states in zip(
            partial_choice_probs_unobserved_states,
            possible_unobserved_states,
            possible_pre_period_unobserved_states,
        ):
            weights = jax.vmap(
                partial_weight_func,
                in_axes=(None, 0, 0),
            )(
                params_in,
                pre_period_unobserved_states,
                unobserved_state_specs["pre_period_choices"][~full_mask],
            )

            unweighted_choice_probs = partial_choice_prob(
                value_in=value_in,
                endog_grid_in=endog_grid_in,
                params_in=params_in,
            )
            objects[i] = {}
            objects[i]["unweighted_choice_probs"] = unweighted_choice_probs
            objects[i]["weights"] = weights

            i += 1
            unobserved_probs += jnp.nan_to_num(
                weights * unweighted_choice_probs, nan=0.0
            )

        choice_probs_final = choice_probs_final.at[unobserved_states_index].set(
            unobserved_probs
        )

        choice_probs_full = partial_choice_probs_full_observed_states(
            value_in=value_in,
            endog_grid_in=endog_grid_in,
            params_in=params_in,
        )

        if weight_full_states:
            weight_choice_probs_full = jax.vmap(
                partial_weight_func,
                in_axes=(None, 0, 0),
            )(
                params_in,
                pre_period_full_observed_states,
                unobserved_state_specs["pre_period_choices"][full_mask],
            )

            choice_probs_final = choice_probs_final.at[observed_states_index].set(
                choice_probs_full * weight_choice_probs_full
            )
        else:
            choice_probs_final = choice_probs_final.at[observed_states_index].set(
                choice_probs_full
            )

        return choice_probs_final

    return choice_prob_func


def create_partial_choice_prob_calculation(
    observed_states,
    observed_choices,
    model,
):
    discrete_observed_state_choice_indexes = get_state_choice_index_per_discrete_state(
        states=observed_states,
        map_state_choice_to_index=model["model_structure"]["map_state_choice_to_index"],
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
        observed_states.pop(second_continuous_state_name)

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
    choice_prob_across_choices, _, _ = calculate_choice_probs_and_unsqueezed_logsum(
        choice_values_per_state=value_per_agent_interp,
        taste_shock_scale=params["lambda"],
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
    state_choice_vec.pop("wealth")

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
    state_choice_vec.pop("wealth")

    value_interp = interp_value_on_wealth(
        wealth=state["wealth"],
        endog_grid=endog_grid_agent,
        value=value_agent,
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
    )

    return value_interp


def calculate_weights_for_each_state(params, state_vec, choice, options, weight_func):
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
    return weight_func(**state_vec, params=params, choice=choice, options=options)
