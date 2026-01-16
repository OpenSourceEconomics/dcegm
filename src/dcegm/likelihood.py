"""THIS MODULE IS NOT TESTED YET.

IT IS WORK IN PROGRESS.

"""

import copy
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from dcegm.backward_induction import backward_induction
from dcegm.egm.aggregate_marginal_utility import (
    calculate_choice_probs_and_unsqueezed_logsum,
)
from dcegm.interfaces.index_functions import get_state_choice_index_per_discrete_states
from dcegm.interfaces.interface import choice_values_for_states
from dcegm.interfaces.jit_large_arrays import (
    merg_non_jit_batch_info_and_jit_batch_info,
    merge_non_jit_and_jit_model_structure,
    split_structure_and_batch_info,
)


def create_individual_likelihood_function(
    income_shock_draws_unscaled,
    income_shock_weights,
    batch_info,
    model_structure,
    model_config,
    model_funcs,
    model_specs,
    observed_states: Dict[str, int],
    observed_choices,
    params_all,
    unobserved_state_specs=None,
    return_model_solution=False,
    use_probability_of_observed_states=True,
    slow_version=False,
):

    choice_prob_func, data_from_observed_states = create_choice_prob_function(
        model_structure=model_structure,
        model_config=model_config,
        model_funcs=model_funcs,
        model_specs=model_specs,
        observed_states=observed_states,
        observed_choices=observed_choices,
        unobserved_state_specs=unobserved_state_specs,
        use_probability_of_observed_states=use_probability_of_observed_states,
        return_weight_func=False,
    )

    (
        model_structure_for_jit,
        batch_info_for_jit,
        model_structure_non_jit,
        batch_info_non_jit,
    ) = split_structure_and_batch_info(model_structure, batch_info)

    def individual_likelihood_to_jit(params, model_structure_jit, batch_info_jit):
        params_update = params_all.copy()
        params_update.update(params)

        # Merge back parts together. The non_jit objects are fixed in the closure.
        model_structure_merged = merge_non_jit_and_jit_model_structure(
            model_structure_jit, model_structure_non_jit
        )
        batch_info_merged = merg_non_jit_batch_info_and_jit_batch_info(
            batch_info_jit, batch_info_non_jit
        )

        value, policy, endog_grid = backward_induction(
            params=params_update,
            income_shock_draws_unscaled=income_shock_draws_unscaled,
            income_shock_weights=income_shock_weights,
            model_config=model_config,
            model_funcs=model_funcs,
            model_structure=model_structure_merged,
            batch_info=batch_info_merged,
        )

        choice_probs = choice_prob_func(
            value_in=value,
            endog_grid_in=endog_grid,
            params_in=params_update,
            data_from_observed=data_from_observed_states,
        )
        # Negative ll contributions are positive numbers. The smaller the better the fit
        # Add high fixed punishment for not explained choices
        neg_likelihood_contributions = (-jnp.log(choice_probs)).clip(max=999)

        if return_model_solution:
            sol_dict = {
                "value": value,
                "policy": policy,
                "endog_grid": endog_grid,
            }
            return neg_likelihood_contributions, sol_dict
        else:
            return neg_likelihood_contributions

    if slow_version:
        likelihood_function_int = individual_likelihood_to_jit
    else:
        likelihood_function_int = jax.jit(individual_likelihood_to_jit)

    def likelihood_function(params):
        return likelihood_function_int(
            params=params,
            model_structure_jit=model_structure_for_jit,
            batch_info_jit=batch_info_for_jit,
        )

    return likelihood_function


def create_choice_prob_function(
    model_structure,
    model_config,
    model_funcs,
    model_specs,
    observed_states,
    observed_choices,
    unobserved_state_specs,
    use_probability_of_observed_states,
    return_weight_func,
):
    if unobserved_state_specs is None:
        choice_prob_func, data_from_observed_states = (
            create_partial_choice_prob_calculation(
                observed_states=observed_states,
                observed_choices=observed_choices,
                model_structure=model_structure,
                model_config=model_config,
                model_funcs=model_funcs,
            )
        )
    else:

        choice_prob_func, data_from_observed_states = (
            create_choice_prob_func_unobserved_states(
                model_structure=model_structure,
                model_config=model_config,
                model_funcs=model_funcs,
                model_specs=model_specs,
                observed_states=observed_states,
                observed_choices=observed_choices,
                unobserved_state_specs=unobserved_state_specs,
                use_probability_of_observed_states=use_probability_of_observed_states,
                return_weight_func=return_weight_func,
            )
        )

    return choice_prob_func, data_from_observed_states


def create_choice_prob_func_unobserved_states(
    model_structure,
    model_config,
    model_funcs,
    model_specs,
    observed_states: Dict[str, int],
    observed_choices,
    unobserved_state_specs,
    use_probability_of_observed_states=True,
    return_weight_func=False,
):

    unobserved_state_names = unobserved_state_specs["observed_bools_states"].keys()
    observed_bools = unobserved_state_specs["observed_bools_states"]

    # Create weighting vars by extracting states and choices
    weighting_vars = unobserved_state_specs["weighting_vars"]

    # Add unobserved states with appendix new and bools indicating if state is observed
    for state_name in unobserved_state_names:
        weighting_vars[state_name + "_new"] = observed_states[state_name]

    # Read out possible values for unobserved states. Two cases: Either they are explicitly defined
    # by the user, in the case of only a subset of values. Or they are all possible values, then it is
    # read out of the specification of the model.
    any_custom_unobserved_states = "custom_unobserved_states" in unobserved_state_specs

    unobserved_state_values = {}
    for state_name in unobserved_state_specs["observed_bools_states"].keys():
        # Check if state is custom defined
        if any_custom_unobserved_states:
            state_custom_defined = (
                state_name in unobserved_state_specs["custom_unobserved_states"]
            )
        else:
            state_custom_defined = False

        if state_custom_defined:
            state_values = unobserved_state_specs["custom_unobserved_states"][
                state_name
            ]
        else:
            if state_name in model_structure["stochastic_states_names"]:
                state_values = model_config["stochastic_states"][state_name]
            else:
                state_values = model_config["deterministic_states"][state_name]
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
            # Same for variables to weight function
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

    observed_weights = jnp.asarray(observed_weights)

    # Create a list of partial choice probability functions for each unique
    # combination of unobserved states.
    partial_choice_probs_unobserved_states = []
    data_for_unobserved_states = []
    for states in possible_states:
        choice_func, data = create_partial_choice_prob_calculation(
            observed_states=states,
            observed_choices=observed_choices,
            model_structure=model_structure,
            model_config=model_config,
            model_funcs=model_funcs,
        )
        partial_choice_probs_unobserved_states.append(choice_func)
        data_for_unobserved_states.append(data)

    partial_weight_func = (
        lambda params_in, weight_vars: calculate_weights_for_each_state(
            params=params_in,
            weight_vars=weight_vars,
            model_specs=model_specs,
            weight_func=unobserved_state_specs["weight_func"],
        )
    )

    n_obs = len(observed_choices)

    # Use jax tree map to make only jax arrays of possible states and weighting vars
    possible_states = jax.tree_util.tree_map(lambda x: jnp.asarray(x), possible_states)
    weighting_vars_for_possible_states = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x), weighting_vars_for_possible_states
    )

    def choice_prob_func(value_in, endog_grid_in, params_in, data_for_choice_funcs):
        choice_probs_final = jnp.zeros(n_obs, dtype=jnp.float64)
        integrate_out_weights = jnp.zeros(n_obs, dtype=jnp.float64)
        for partial_choice_prob, data_for_choice_func, weighting_vars in zip(
            partial_choice_probs_unobserved_states,
            data_for_choice_funcs,
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
                data_from_observed=data_for_choice_func,
            )

            weighted_choice_prob = jnp.nan_to_num(
                unweighted_choice_probs * unobserved_weights * observed_weights, nan=0.0
            )

            integrate_out_weights += unobserved_weights * observed_weights

            choice_probs_final += weighted_choice_prob

        if not use_probability_of_observed_states:
            choice_probs_final /= integrate_out_weights

        return choice_probs_final

    def weight_only_func(params_in):
        weights = np.zeros((n_obs, len(possible_states)), dtype=np.float64)
        count = 0
        for weighting_vars in weighting_vars_for_possible_states:
            unobserved_weights = jax.vmap(
                partial_weight_func,
                in_axes=(None, 0),
            )(
                params_in,
                weighting_vars,
            )

            weights[:, count] = unobserved_weights
            count += 1
        return (
            weights,
            observed_weights,
            possible_states,
            weighting_vars_for_possible_states,
        )

    if return_weight_func:
        return choice_prob_func, weight_only_func, data_for_unobserved_states
    else:
        return choice_prob_func, data_for_unobserved_states


def create_partial_choice_prob_calculation(
    observed_states,
    observed_choices,
    model_structure,
    model_config,
    model_funcs,
):
    discrete_observed_state_choice_indexes = get_state_choice_index_per_discrete_states(
        states=observed_states,
        map_state_choice_to_index=model_structure[
            "map_state_choice_to_index_with_proxy"
        ],
        discrete_states_names=model_structure["discrete_states_names"],
    )

    data_from_observed_wrapped = (
        observed_states,
        observed_choices,
        discrete_observed_state_choice_indexes,
    )

    def partial_choice_prob_func(
        value_in, endog_grid_in, params_in, data_from_observed
    ):
        return calc_choice_prob_for_state_choices(
            value_solved=value_in,
            endog_grid_solved=endog_grid_in,
            params=params_in,
            states=data_from_observed[0],
            choices=data_from_observed[1],
            state_choice_indexes=data_from_observed[2],
            model_config=model_config,
            model_funcs=model_funcs,
        )

    return partial_choice_prob_func, data_from_observed_wrapped


def calc_choice_prob_for_state_choices(
    value_solved,
    endog_grid_solved,
    params,
    states,
    choices,
    state_choice_indexes,
    model_config,
    model_funcs,
):
    """This function interpolates the policy and value function for all agents.

    It uses the states at the beginning of period to select the solved policy and value
    and then interpolates the wealth at the beginning of period on them.

    """

    choice_prob_across_choices = calc_choice_probs_for_states(
        value_solved=value_solved,
        endog_grid_solved=endog_grid_solved,
        state_choice_indexes=state_choice_indexes,
        params=params,
        states=states,
        model_config=model_config,
        model_funcs=model_funcs,
    )
    choice_probs = jnp.take_along_axis(
        choice_prob_across_choices, choices[:, None], axis=1
    )[:, 0]
    return choice_probs


def calc_choice_probs_for_states(
    value_solved,
    endog_grid_solved,
    state_choice_indexes,
    params,
    states,
    model_config,
    model_funcs,
):
    choice_values_per_state = choice_values_for_states(
        value_solved=value_solved,
        endog_grid_solved=endog_grid_solved,
        state_choice_indexes=state_choice_indexes,
        params=params,
        states=states,
        model_config=model_config,
        model_funcs=model_funcs,
    )

    if model_funcs["taste_shock_function"]["taste_shock_scale_is_scalar"]:
        taste_shock_scale = model_funcs["taste_shock_function"][
            "read_out_taste_shock_scale"
        ](params)
    else:
        taste_shock_scale_per_state_func = model_funcs["taste_shock_function"][
            "taste_shock_scale_per_state"
        ]
        taste_shock_scale = vmap(taste_shock_scale_per_state_func, in_axes=(0, None))(
            states, params
        )
        taste_shock_scale = taste_shock_scale[:, None]

    choice_prob_across_choices, _, _ = calculate_choice_probs_and_unsqueezed_logsum(
        choice_values_per_state=choice_values_per_state,
        taste_shock_scale=taste_shock_scale,
    )
    return choice_prob_across_choices


def calculate_weights_for_each_state(params, weight_vars, model_specs, weight_func):
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
    return weight_func(**weight_vars, params=params, model_specs=model_specs)
