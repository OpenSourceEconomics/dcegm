"""Interface for the DC-EGM algorithm."""

from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax.lax
import jax.numpy as jnp
import numpy as np

from dcegm.final_periods import solve_last_two_periods
from dcegm.law_of_motion import calc_cont_grids_next_period
from dcegm.solve_single_period import solve_single_period


def backward_induction(
    params: Dict[str, float],
    income_shock_draws_unscaled: jnp.ndarray,
    income_shock_weights: jnp.ndarray,
    model_config: Dict[str, Any],
    model_funcs: Dict[str, Callable],
    model_structure: Dict[str, Any],
    batch_info: Dict[str, Any],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Do backward induction and solve for optimal policy and value function.

    Args:
        params (dict): Dictionary containing the model parameters.
        options (dict): Dictionary containing the model options.
        period_specific_state_objects (np.ndarray): Dictionary containing
            period-specific state and state-choice objects, with the following keys:
            - "state_choice_mat" (jnp.ndarray)
            - "idx_state_of_state_choice" (jnp.ndarray)
            - "reshape_state_choice_vec_to_mat" (callable)
            - "transform_between_state_and_state_choice_vec" (callable)
        exog_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the exogenous savings grid.
        has_second_continuous_state (bool): Boolean indicating whether the model
            features a second continuous state variable. If False, the only
            continuous state variable is consumption/savings.
        state_space (np.ndarray): 2d array of shape (n_states, n_state_variables + 1)
            which serves as a collection of all possible states. By convention,
            the first column must contain the period and the last column the
            exogenous processes. Any other state variables are in between.
            E.g. if the two state variables are period and lagged choice and all choices
            are admissible in each period, the shape of the state space array is
            (n_periods * n_choices, 3).
        state_choice_space (np.ndarray): 2d array of shape
            (n_feasible_states, n_state_and_exog_variables + 1) containing all
            feasible state-choice combinations. By convention, the second to last
            column contains the exogenous process. The last column always contains the
            choice to be made (which is not a state variable).
        income_shock_draws_unscaled (np.ndarray): 1d array of shape (n_quad_points,)
            containing the Hermite quadrature points unscaled.
        income_shock_weights (np.ndarrray): 1d array of shape
            (n_stochastic_quad_points) with weights for each stoachstic shock draw.
        n_periods (int): Number of periods.
        model_funcs (dict): Dictionary containing following model functions:
            - compute_marginal_utility (callable): User-defined function to compute the
                agent's marginal utility. The input ```params``` is already partialled
                in.
            - compute_inverse_marginal_utility (Callable): Function for calculating the
                inverse marginal utiFality, which takes the marginal utility as only
                 input.
            - compute_next_period_wealth (callable): User-defined function to compute
                the agent's wealth of the next period (t + 1). The inputs
                ```saving```, ```shock```, ```params``` and ```options```
                are already partialled in.
            - transition_vector_by_state (Callable): Partialled transition function
                return transition vector for each state.
            - final_period_partial (Callable): Partialled function for calculating the
                consumption as well as value function and marginal utility in the final
                period.
        compute_upper_envelope (Callable): Function for calculating the upper
                envelope of the policy and value function. If the number of discrete
                choices is 1, this function is a dummy function that returns the policy
                and value function as is, without performing a fast upper envelope
                scan.

    Returns:
        dict: Dictionary containing the period-specific endog_grid, policy, and value
            from the backward induction.

    """
    continuous_states_info = model_config["continuous_states_info"]

    cont_grids_next_period = calc_cont_grids_next_period(
        model_structure=model_structure,
        model_config=model_config,
        income_shock_draws_unscaled=income_shock_draws_unscaled,
        params=params,
        model_funcs=model_funcs,
    )

    # Create solution containers. The 20 percent extra in wealth grid needs to go
    # into tuning parameters
    n_total_wealth_grid = model_config["tuning_params"]["n_total_wealth_grid"]
    (
        value_solved,
        policy_solved,
        endog_grid_solved,
    ) = create_solution_container(
        model_config=model_config,
        model_structure=model_structure,
    )

    # Solve the last two periods. We do this separately as the marginal utility of
    # the child states in the last period is calculated from the marginal utility
    # function of the bequest function, which might differ.
    (
        value_solved,
        policy_solved,
        endog_grid_solved,
    ) = solve_last_two_periods(
        params=params,
        continuous_states_info=continuous_states_info,
        cont_grids_next_period=cont_grids_next_period,
        income_shock_weights=income_shock_weights,
        model_funcs=model_funcs,
        last_two_period_batch_info=batch_info["last_two_period_info"],
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
    )

    # If it is a two period model we are done.
    if batch_info["two_period_model"]:
        return value_solved, policy_solved, endog_grid_solved

    def partial_single_period(carry, xs):
        return solve_single_period(
            carry=carry,
            xs=xs,
            params=params,
            continuous_grids_info=continuous_states_info,
            cont_grids_next_period=cont_grids_next_period,
            model_funcs=model_funcs,
            income_shock_weights=income_shock_weights,
        )

    for id_segment in range(batch_info["n_segments"]):
        segment_info = batch_info[f"batches_info_segment_{id_segment}"]

        carry_start = (
            value_solved,
            policy_solved,
            endog_grid_solved,
        )

        final_carry, _ = jax.lax.scan(
            f=partial_single_period,
            init=carry_start,
            xs=(
                segment_info["batches_state_choice_idx"],
                segment_info["child_state_choices_to_aggr_choice"],
                segment_info["child_states_to_integrate_stochastic"],
                segment_info["child_state_choice_idxs_to_interp"],
                segment_info["child_states_idxs"],
                segment_info["state_choices"],
                segment_info["state_choices_childs"],
            ),
        )

        if not segment_info["batches_cover_all"]:
            last_batch_info = segment_info["last_batch_info"]
            extra_final_carry, () = partial_single_period(
                carry=final_carry,
                xs=(
                    last_batch_info["state_choice_idx"],
                    last_batch_info["child_state_choices_to_aggr_choice"],
                    last_batch_info["child_states_to_integrate_stochastic"],
                    last_batch_info["child_state_choice_idxs_to_interp"],
                    last_batch_info["child_states_idxs"],
                    last_batch_info["state_choices"],
                    last_batch_info["state_choices_childs"],
                ),
            )

            (
                value_solved,
                policy_solved,
                endog_grid_solved,
            ) = extra_final_carry
        else:
            (
                value_solved,
                policy_solved,
                endog_grid_solved,
            ) = final_carry

    return (
        value_solved,
        policy_solved,
        endog_grid_solved,
    )


def create_solution_container(
    model_config: Dict[str, Any],
    model_structure: Dict[str, Any],
):
    """Create solution containers for value, policy, and endog_grid."""

    # Read out grid size
    n_total_wealth_grid = model_config["tuning_params"]["n_total_wealth_grid"]
    n_state_choices = model_structure["state_choice_space"].shape[0]

    # Check if second continuous state exists and read out array size
    continuous_states_info = model_config["continuous_states_info"]
    if continuous_states_info["second_continuous_exists"]:
        n_second_continuous_grid = continuous_states_info["n_second_continuous_grid"]

        value_solved = jnp.full(
            (n_state_choices, n_second_continuous_grid, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
        policy_solved = jnp.full(
            (n_state_choices, n_second_continuous_grid, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
        endog_grid_solved = jnp.full(
            (n_state_choices, n_second_continuous_grid, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
    else:
        value_solved = jnp.full(
            (n_state_choices, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
        policy_solved = jnp.full(
            (n_state_choices, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
        endog_grid_solved = jnp.full(
            (n_state_choices, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )

    return value_solved, policy_solved, endog_grid_solved
