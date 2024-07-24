"""THIS MODULE IS NOT TESTED YET.

IT IS WORK IN PROGRESS.

"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from dcegm.egm.aggregate_marginal_utility import (
    calculate_choice_probs_and_unsqueezed_logsum,
)
from dcegm.interface import get_state_choice_index_per_state
from dcegm.interpolation import interp_value_on_wealth
from dcegm.solve import get_solve_func_for_model


def create_individual_likelihood_function_for_model(
    model: Dict[str, Any],
    observed_states: Dict[str, int],
    observed_wealth: np.array,
    observed_choices: np.array,
    params_all,
    unobserved_state_specs=None,
):
    solve_func = get_solve_func_for_model(
        model=model,
    )

    if unobserved_state_specs is not None:
        full_mask = unobserved_state_specs["observed_bool"]
        full_observed_states = {
            name: observed_states[name][full_mask]
            for name in model["model_structure"]["state_space_names"]
        }
        full_observed_choices = observed_choices[full_mask]
        full_observed_wealth = observed_wealth[full_mask]
    else:
        full_observed_states = observed_states
        full_observed_choices = observed_choices
        full_observed_wealth = observed_wealth

    # Create the calculation of the choice probabilities, which takes parameters as
    # input as well as the solved endogenous wealth grid and the values. We first
    # create it for the fully observed states.
    partial_choice_probs_full_observed_states = create_partial_choice_prob_calculation(
        observed_states=full_observed_states,
        observed_choices=full_observed_choices,
        observed_wealth=full_observed_wealth,
        model=model,
    )

    if unobserved_state_specs is not None:
        unobserved_state_values = []
        for state_name in unobserved_state_specs["states"]:
            if state_name in model["model_structure"]["exog_states_names"]:
                state_values = model["options"]["state_space"]["exogenous_processes"][
                    state_name
                ]["states"]
            else:
                state_values = model["options"]["state_space"]["endogenous_states"][
                    state_name
                ]

    else:
        # If all states are fully observed, the choice probability function
        # corresponds to the one for the fully observed states.
        choice_prob_func = partial_choice_probs_full_observed_states

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
        ).clip(min=1e-10)
        likelihood_contributions = jnp.log(choice_probs)
        log_value = jnp.sum(-likelihood_contributions)
        return log_value, likelihood_contributions

    return jax.jit(individual_likelihood)


def create_partial_choice_prob_calculation(
    observed_states,
    observed_choices,
    observed_wealth,
    model,
):
    observed_state_choice_indexes = get_state_choice_index_per_state(
        states=observed_states,
        map_state_choice_to_index=model["model_structure"]["map_state_choice_to_index"],
        state_space_names=model["model_structure"]["state_space_names"],
    )

    options = model["options"]

    def partial_choice_prob_func(value_in, endog_grid_in, params_in):
        return calc_choice_prob_for_state_choices(
            value_solved=value_in,
            endog_grid_solved=endog_grid_in,
            params=params_in,
            states=observed_states,
            choices=observed_choices,
            state_choice_indexes=observed_state_choice_indexes,
            oberseved_wealth=observed_wealth,
            choice_range=np.arange(options["model_params"]["n_choices"], dtype=int),
            compute_utility=model["model_funcs"]["compute_utility"],
        )

    return partial_choice_prob_func


def calc_choice_prob_for_state_choices(
    value_solved,
    endog_grid_solved,
    params,
    states,
    choices,
    state_choice_indexes,
    oberseved_wealth,
    choice_range,
    compute_utility,
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
        oberseved_wealth=oberseved_wealth,
        choice_range=choice_range,
        compute_utility=compute_utility,
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
    oberseved_wealth,
    choice_range,
    compute_utility,
):
    value_grid_agent = jnp.take(
        value_solved, state_choice_indexes, axis=0, mode="fill", fill_value=jnp.nan
    )
    endog_grid_agent = jnp.take(endog_grid_solved, state_choice_indexes, axis=0)
    vectorized_interp = jax.vmap(
        jax.vmap(
            interpolate_value_for_state_in_each_choice,
            in_axes=(None, None, 0, 0, 0, None, None),
        ),
        in_axes=(0, 0, 0, 0, None, None, None),
    )

    value_per_agent_interp = vectorized_interp(
        observed_states,
        oberseved_wealth,
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


def interpolate_value_for_state_in_each_choice(
    state,
    resource_at_beginning_of_period,
    endog_grid_agent,
    value_agent,
    choice,
    params,
    compute_utility,
):
    state_choice_vec = {**state, "choice": choice}

    value_interp = interp_value_on_wealth(
        wealth=resource_at_beginning_of_period,
        endog_grid=endog_grid_agent,
        value=value_agent,
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
    )

    return value_interp
