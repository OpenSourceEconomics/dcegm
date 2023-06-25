"""Wrapper function to solve the final period of the model."""
from typing import Callable
from typing import Tuple

import jax.numpy as jnp
from jax import vmap


def final_period_wrapper(
    final_period_choice_states: jnp.ndarray,
    final_period_solution_partial: Callable,
    sum_state_choices_to_state,
    state_times_state_choice_mat: jnp.ndarray,
    resources_last_period: jnp.ndarray,
    taste_shock_scale: float,
    income_shock_draws: jnp.ndarray,
    income_shock_weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes solution to final period for policy and value function.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        final_period_choice_states (np.ndarray): Collection of all possible state-choice
            combinations in the final period.
        taste_shock_scale (float): The taste shock scale.
        income_shock_draws (np.ndarray): 1d array of shape (n_quad_points,) containing
            the Hermite quadrature points.
        income_shock_weights (np.ndarrray): 1d array of shape (n_stochastic_quad_points)
            with weights for each stoachstic shock draw.

    Returns:
        tuple:

        - endog_grid_final (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the
            endogenous wealth grid for all final states, choices, and
            end of period assets from the period before.
        - final_policy (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the optimal
            policy for all final states, choices, end of period assets, and
            income shocks.
        - final_value (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, choices, end of period assets, and
            income shocks.
        - marginal_utilities_choices (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, choices, end of period assets, and
            income shocks.
        - max_exp_values (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, choices, end of period assets, and
            income shocks.

    """

    # Compute for each wealth grid point the optimal policy and value function as well
    # as the marginal utility of consumption for all choices.
    final_policy, final_value, marginal_utilities_choices = vmap(
        vmap(
            vmap(
                final_period_solution_partial,
                in_axes=(None, 0, None),
            ),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, 0, 0),
    )(
        final_period_choice_states[:, :-1],
        resources_last_period,
        final_period_choice_states[:, -1],
    )

    return (
        resources_last_period,
        final_policy,
        final_value,
        marginal_utilities_choices,
    )


def aggregate_marg_utils_exp_values(
    final_value_state_choice: jnp.ndarray,
    state_times_state_choice_mat: jnp.ndarray,
    marg_util_state_choice: jnp.ndarray,
    sum_state_choices_to_state: jnp.ndarray,
    taste_shock_scale: float,
    income_shock_weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the aggregate marginal utilities and expected values.

    Args:
        final_value_state_choice (jnp.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the value function for all
            states, choices, and income shocks.
        marg_util_state_choice (jnp.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the marginal utility of
            consumption for all states, choices, and income shocks.
        sum_state_choices_to_state (jnp.ndarray): 2d array of shape
            (n_states, n_states * n_choices) with state_space size  with ones, where
            state-choice belongs to state.
        taste_shock_scale (float): The taste shock scale.
        income_shock_weights (jnp.ndarray): 1d array of shape (n_stochastic_quad_points)

    Returns:
        tuple:

        - marginal_utils_draws (np.ndarray): 1d array of shape
        (n_states, n_savings, n_income_shocks,) of the aggregate marginal utilities.
        - max_exp_values_draws (np.ndarray): 1d array of shape
        (n_states, n_savings, n_income_shocks,) of the aggregate expected values.

    """
    max_value_per_state = jnp.take(
        final_value_state_choice,
        state_times_state_choice_mat,
        axis=0,
    ).max(axis=1)
    rescale_value = jnp.tensordot(
        sum_state_choices_to_state, max_value_per_state, axes=(0, 0)
    )

    exp_value = jnp.exp((final_value_state_choice - rescale_value)) ** (
        1 / taste_shock_scale
    )
    sum_exp = jnp.tensordot(sum_state_choices_to_state, exp_value, axes=(1, 0))

    max_exp_values_draws = max_value_per_state + taste_shock_scale * jnp.log(sum_exp)
    marginal_utils_draws = jnp.divide(
        jnp.tensordot(
            sum_state_choices_to_state,
            jnp.multiply(exp_value, marg_util_state_choice),
            axes=(1, 0),
        ),
        sum_exp,
    )
    return (
        marginal_utils_draws @ income_shock_weights,
        max_exp_values_draws @ income_shock_weights,
    )
