from typing import Any, Dict

from jax import numpy as jnp


def create_solution_container(
    n_total_wealth_grid: int,
    n_state_choices: int,
    n_continuous_state_combinations: int,
    store_endog_grid: bool,
):
    """Create solution containers for value, policy, and endog_grid.

    endog_grid is only allocated when store_endog_grid is True. When the
    Druedahl-Jorgensen upper envelope is used, the "endogenous" grid is actually the
    fixed exogenous grid, so storing it is skipped and callers read the exogenous
    grid directly instead: each reader recomputes the state-choice's own grid on
    demand (see compute_own_dj_wealth_grid).

    """
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
    endog_grid_solved = (
        jnp.full(
            (n_state_choices, n_continuous_state_combinations, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
        if store_endog_grid
        else None
    )

    return value_solved, policy_solved, endog_grid_solved
