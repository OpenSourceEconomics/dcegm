from typing import Callable
from typing import Dict

import jax
import jax.numpy as jnp
import pandas as pd
from dcegm.egm.aggregate_marginal_utility import (
    calculate_choice_probs_and_unsqueezed_logsum,
)
from dcegm.egm.interpolate_marginal_utility import (
    interp_value_and_check_creditconstraint,
)
from dcegm.interpolation import get_index_high_and_low
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.simulation.sim_utils import get_state_choice_index_per_state


def create_likelihood_informations(
    observed_data: pd.DataFrame,
    options: Dict[str, int],
    exog_savings_grid: jnp.ndarray,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    utility_functions_final_period: Dict[str, Callable],
):
    options["state_space"]["n_periods"]

    # ToDo: Make interface with several draw possibilities.
    # ToDo: Some day make user supplied draw function.
    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        options["model_params"]["quadrature_points_stochastic"]
    )

    (
        model_funcs,
        compute_upper_envelope,
        get_state_specific_choice_set,
        update_endog_state_by_state_and_choice,
    ) = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    (
        period_specific_state_objects,
        state_space,
        state_space_names,
        map_state_choice_to_index,
        exog_mapping,
    ) = create_state_space_and_choice_objects(
        options=options,
        get_state_specific_choice_set=get_state_specific_choice_set,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )


def interp_value(
    observed_states,
    state_choice_indexes,
    oberseved_wealth,
    value_solved,
    endog_grid_solved,
    map_state_choice_to_index,
    choice_range,
    params,
    state_space_names,
    compute_utility,
):
    """This function interpolates the policy and value function for all agents.

    It uses the states at the beginning of period to select the solved policy and value
    and then interpolates the wealth at the beginning of period on them.

    """
    state_choice_indexes = get_state_choice_index_per_state(
        map_state_choice_to_index=map_state_choice_to_index,
        states=observed_states,
        state_space_names=state_space_names,
    )

    value_grid_agent = jnp.take(
        value_solved, state_choice_indexes, axis=0, mode="fill", fill_value=jnp.nan
    )
    endog_grid_agent = jnp.take(endog_grid_solved, state_choice_indexes, axis=0)
    vectorized_interp = jax.vmap(
        jax.vmap(
            interpolate_value_and_calc_choice_probabilities,
            in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
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
    choice_probs, _, _ = calculate_choice_probs_and_unsqueezed_logsum(
        choice_values_per_state=value_per_agent_interp,
        taste_shock_scale=params["lambda"],
    )
    return choice_probs


def interpolate_value_and_calc_choice_probabilities(
    resources_beginning_of_period,
    state,
    endog_grid_agent,
    value_agent,
    choice,
    params,
    compute_utility,
):
    ind_high, ind_low = get_index_high_and_low(
        x=endog_grid_agent, x_new=resources_beginning_of_period
    )
    state_choice_vec = {**state, "choice": choice}
    policy_interp, value_interp = interp_value_and_check_creditconstraint(
        value_high=value_agent[ind_high],
        wealth_high=endog_grid_agent[ind_high],
        value_low=value_agent[ind_low],
        wealth_low=endog_grid_agent[ind_low],
        new_wealth=resources_beginning_of_period,
        compute_utility=compute_utility,
        endog_grid_min=endog_grid_agent[1],
        value_at_zero_wealth=value_agent[0],
        state_choice_vec=state_choice_vec,
        params=params,
    )

    return value_interp
