"""Wrapper to solve the final period of the model."""

from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from jax import vmap

from dcegm.solve_single_period import solve_for_interpolated_values


def solve_last_two_periods(
    wealth_and_continuous_state_next_period: jnp.ndarray,
    params: Dict[str, float],
    taste_shock_scale: float,
    income_shock_weights: jnp.ndarray,
    exog_grids: Dict[str, jnp.ndarray],
    wealth_beginning_at_regular: jnp.ndarray,
    model_funcs: Dict[str, Callable],
    batch_info,
    value_solved,
    policy_solved,
    endog_grid_solved,
    has_second_continuous_state: bool,
):
    """Solves the last two periods of the model.

    The last two periods are solved using the EGM algorithm. The last period is
    solved using the user-specified utility function and the second to last period
    is solved using the user-specified utility function and the user-specified
    bequest function.
    Args:
        resources_beginning_of_period (np.ndarray): 2d array of shape
            (n_states, n_grid_wealth) of the resources at the beginning of the
            period.
        params (dict): Dictionary of model parameters.
        compute_utility (callable): User supplied utility function.
        compute_marginal_utility (callable): User supplied marginal utility
            function.
        batch_info (dict): Dictionary containing information about the batch
            size and the state space.
        value_solved (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the value function for
            all states, end of period assets, and income shocks.
        endog_grid_solved (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the endogenous grid
            for all states, end of period assets, and income shocks.

    """
    (
        value_solved,
        policy_solved,
        endog_grid_solved,
        value_interp_final_period,
        marginal_utility_final_last_period,
    ) = solve_final_period(
        idx_state_choices_final_period=batch_info["idx_state_choices_final_period"],
        idx_parent_states_final_period=batch_info["idxs_parent_states_final_period"],
        state_choice_mat_final_period=batch_info["state_choice_mat_final_period"],
        wealth_and_continuous_state_next_period=wealth_and_continuous_state_next_period,
        wealth_beginning_at_regular_period=wealth_beginning_at_regular,
        params=params,
        compute_utility=model_funcs["compute_utility_final"],
        compute_marginal_utility=model_funcs["compute_marginal_utility_final"],
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        has_second_continuous_state=has_second_continuous_state,
    )

    endog_grid, policy, value = solve_for_interpolated_values(
        value_interpolated=value_interp_final_period,
        marginal_utility_interpolated=marginal_utility_final_last_period,
        state_choice_mat=batch_info["state_choice_mat_second_last_period"],
        child_state_idxs=batch_info["child_states_second_last_period"],
        states_to_choices_child_states=batch_info["state_to_choices_final_period"],
        params=params,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
        exog_savings_grid=exog_grids["wealth"],
        model_funcs=model_funcs,
        has_second_continuous_state=has_second_continuous_state,
    )

    idx_second_last = batch_info["idx_state_choices_second_last_period"]

    value_solved = value_solved.at[idx_second_last, ...].set(value)
    policy_solved = policy_solved.at[idx_second_last, ...].set(policy)
    endog_grid_solved = endog_grid_solved.at[idx_second_last, ...].set(endog_grid)

    return value_solved, policy_solved, endog_grid_solved


def solve_final_period(
    idx_state_choices_final_period,
    idx_parent_states_final_period,
    state_choice_mat_final_period,
    wealth_and_continuous_state_next_period: jnp.ndarray,
    wealth_beginning_at_regular_period: Dict[str, jnp.ndarray],
    params: Dict[str, float],
    compute_utility: Callable,
    compute_marginal_utility: Callable,
    value_solved,
    policy_solved,
    endog_grid_solved,
    has_second_continuous_state: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes solution to final period for policy and value function.
    In the last period, everything is consumed, i.e. consumption = savings.
    Args:


    Returns:
        tuple:
        - marginal_utilities_choices (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the marginal utility of
            consumption for all final states, end of period assets, and
            income shocks.
        - final_value (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, end of period assets, and
            income shocks.
        - final_policy (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the optimal
            policy for all final states, end of period assets, and
            income shocks.
    """

    if has_second_continuous_state:
        continuous_state, resources = wealth_and_continuous_state_next_period

        resources = resources[idx_parent_states_final_period]
        continuous_state = continuous_state[idx_parent_states_final_period]
        n_wealth = resources.shape[2]

        value, marg_util = vmap(
            vmap(
                vmap(
                    vmap(
                        calculate_value_and_marg_util_for_each_gridpoint,
                        in_axes=(None, 0, None, None, None),  # income shocks
                    ),
                    in_axes=(None, 0, None, None, None),  # wealth
                ),
                in_axes=(None, 0, None, None, None),  # continuous_state
            ),
            in_axes=(0, 0, None, None, None),  # discrete state choices
        )(
            state_choice_mat_final_period,
            resources,
            params,
            compute_utility,
            compute_marginal_utility,
        )

        resources_regular = wealth_beginning_at_regular_period[
            idx_parent_states_final_period
        ]

        value_regular, _ = vmap(
            vmap(
                vmap(
                    vmap(
                        calculate_value_and_marg_util_for_each_gridpoint,
                        in_axes=(None, 0, None, None, None),  # income shocks
                    ),
                    in_axes=(None, 0, None, None, None),  # wealth
                ),
                in_axes=(None, 0, None, None, None),  # continuous_state
            ),
            in_axes=(0, 0, None, None, None),  # discrete state choices
        )(
            state_choice_mat_final_period,
            resources_regular,
            params,
            compute_utility,
            compute_marginal_utility,
        )

        # Choose which draw we take for policy and value function as those are not
        # saved with respect to the draws
        middle_of_draws = int(value_regular.shape[3] + 1 / 2)
        # Select solutions to store
        value_final = value_regular[:, :, :, middle_of_draws]
        # The policy in the last period is eat it all. Either as bequest or by consuming.
        # The user defines this by the bequest functions. So we save the resources also
        # in the policy container. We also need to sort the resources and value
        resources_to_save = resources_regular[:, :, :, middle_of_draws]
        sort_idx = jnp.argsort(resources_to_save, axis=2)
        resources_sorted = jnp.take_along_axis(resources_to_save, sort_idx, axis=2)
        values_sorted = jnp.take_along_axis(value_final, sort_idx, axis=2)

        # Store results and add zero entry for the first column
        zeros_to_append = jnp.zeros(value_final.shape[:-1])
        # [:, None].repeat(6, axis=1)
        # Add as first column to the sorted arrays
        # values_with_zeros = jnp.stack((zeros_to_append, values_sorted), axis=2)

        # Stack along the second-to-last axis (axis 1)
        values_with_zeros = jnp.concatenate(
            (zeros_to_append[..., None], values_sorted), axis=2
        )
        resources_with_zeros = jnp.concatenate(
            (zeros_to_append[..., None], resources_sorted), axis=2
        )

        value_solved = value_solved.at[
            idx_state_choices_final_period, :, : n_wealth + 1
        ].set(values_with_zeros)
        policy_solved = policy_solved.at[
            idx_state_choices_final_period, :, : n_wealth + 1
        ].set(resources_with_zeros)
        endog_grid_solved = endog_grid_solved.at[
            idx_state_choices_final_period, :, : n_wealth + 1
        ].set(resources_with_zeros)

    else:
        resources = wealth_and_continuous_state_next_period

        resources = resources[idx_parent_states_final_period]
        n_wealth = resources.shape[1]

        value, marg_util = vmap(
            vmap(
                vmap(
                    calculate_value_and_marg_util_for_each_gridpoint,
                    in_axes=(None, 0, None, None, None),  # income shocks
                ),
                in_axes=(None, 0, None, None, None),  # wealth
            ),
            in_axes=(0, 0, None, None, None),  # discrete state choices
        )(
            state_choice_mat_final_period,
            resources,
            params,
            compute_utility,
            compute_marginal_utility,
        )
        # Choose which draw we take for policy and value function as those are not
        # saved with respect to the draws
        middle_of_draws = int(value.shape[2] + 1 / 2)
        # Select solutions to store
        value_final = value[:, :, middle_of_draws]
        # The policy in the last period is eat it all. Either as bequest or by consuming.
        # The user defines this by the bequest functions. So we save the resources also
        # in the policy container. We also need to sort the resources and value
        resources_to_save = resources[:, :, middle_of_draws]
        sort_idx = jnp.argsort(resources_to_save, axis=1)
        resources_sorted = jnp.take_along_axis(resources_to_save, sort_idx, axis=1)
        values_sorted = jnp.take_along_axis(value_final, sort_idx, axis=1)

        # Store results and add zero entry for the first column
        zeros_to_append = jnp.zeros(value_final.shape[0])
        # Add as first column to the sorted arrays
        values_with_zeros = jnp.column_stack((zeros_to_append, values_sorted))
        resources_with_zeros = jnp.column_stack((zeros_to_append, resources_sorted))

        value_solved = value_solved.at[
            idx_state_choices_final_period, : n_wealth + 1
        ].set(values_with_zeros)
        policy_solved = policy_solved.at[
            idx_state_choices_final_period, : n_wealth + 1
        ].set(resources_with_zeros)
        endog_grid_solved = endog_grid_solved.at[
            idx_state_choices_final_period, : n_wealth + 1
        ].set(resources_with_zeros)

    return (
        value_solved,
        policy_solved,
        endog_grid_solved,
        value,
        marg_util,
    )


def calculate_value_and_marg_util_for_each_gridpoint(
    state_choice_vec, resources, params, compute_utility, compute_marginal_utility
):
    """Continuous state is missing here!"""
    value = compute_utility(
        **state_choice_vec,
        resources=resources,
        params=params,
    )

    marg_util = compute_marginal_utility(
        **state_choice_vec,
        resources=resources,
        params=params,
    )

    return value, marg_util
