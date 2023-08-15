"""Wrapper function to solve the final period of the model."""
from typing import Callable
from typing import Tuple

import jax.numpy as jnp
from jax import vmap


def solve_final_period(
    final_period_choice_states: jnp.ndarray,
    final_period_solution_partial: Callable,
    resources_last_period: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes solution to final period for policy and value function.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        final_period_choice_states (np.ndarray): Collection of all possible state-choice
            combinations in the final period.

    Returns:
        tuple:


        - final_value (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, end of period assets, and
            income shocks.
        - final_policy (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the optimal
            policy for all final states, end of period assets, and
            income shocks.
        - marginal_utilities_choices (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the marginal utility of
            consumption for all final states, end of period assets, and
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
        final_value,
        final_policy,
        marginal_utilities_choices,
    )
