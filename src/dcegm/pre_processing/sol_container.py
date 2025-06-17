from typing import Any, Dict

from jax import numpy as jnp


def create_solution_container(
    model_config: Dict[str, Any],
    n_state_choices: int,
):
    """Create solution containers for value, policy, and endog_grid."""

    # Read out grid size
    n_total_wealth_grid = model_config["tuning_params"]["n_total_wealth_grid"]

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
