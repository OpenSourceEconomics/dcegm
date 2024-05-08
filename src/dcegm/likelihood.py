"""THIS MODULE IS NOT TESTED YET.

IT IS WORK IN PROGRESS.

"""
from typing import Any
from typing import Callable
from typing import Dict

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
    options: Dict[str, Any],
    observed_states: Dict[str, int],
    observed_wealth: np.array,
    observed_choices: np.array,
    exog_savings_grid: np.ndarray,
    params_all,
):
    solve_func = get_solve_func_for_model(
        model=model, exog_savings_grid=exog_savings_grid, options=options
    )

    observed_state_choice_indexes = get_state_choice_index_per_state(
        states=observed_states,
        map_state_choice_to_index=model["model_structure"]["map_state_choice_to_index"],
        state_space_names=model["model_structure"]["state_space_names"],
    )

    # Create the calculation of the choice probabilities, which takes parameters as
    # input as well as the solved endogenous wealth grid and the values.
    def partial_choice_prob_calculation(value_in, endog_grid_in, params_in):
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

    def individual_likelihood(params):
        params_update = params_all.copy()
        params_update.update(params)

        (
            value_solved,
            policy_solved,
            endog_grid_solved,
        ) = solve_func(params_update)
        choice_probs = partial_choice_prob_calculation(
            value_in=value_solved,
            endog_grid_in=endog_grid_solved,
            params_in=params_update,
        ).clip(min=1e-10)
        likelihood_contributions = jnp.log(choice_probs)
        log_value = jnp.sum(-likelihood_contributions)
        return log_value, likelihood_contributions

    return jax.jit(individual_likelihood)


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
