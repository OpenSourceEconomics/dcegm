from typing import Any, Dict

from jax import numpy as jnp


def create_solution_container(
    n_total_wealth_grid: int,
    n_state_choices: int,
    n_continuous_state_combinations: int,
):
    """Create solution containers for value, policy, and endog_grid."""
    value_solved = jnp.full(
        (n_state_choices, n_continuous_state_combinations, n_total_wealth_grid),
        dtype=jnp.float64,
        fill_value=jnp.nan,
    )
    policy_solved = jnp.full(
        (n_state_choices, n_continuous_state_combinations, n_total_wealth_grid),
        dtype=jnp.float64,
        fill_value=jnp.nan,
    )
    endog_grid_solved = jnp.full(
        (n_state_choices, n_continuous_state_combinations, n_total_wealth_grid),
        dtype=jnp.float64,
        fill_value=jnp.nan,
    )

    return value_solved, policy_solved, endog_grid_solved
