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
    grid directly instead (see model_config["continuous_states_info"]["dj_wealth_grid"]).

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


def broadcast_dj_wealth_grid(continuous_states_info: Dict[str, Any], shape):
    """Broadcast the fixed Druedahl-Jorgensen wealth grid to the given shape.

    Used in place of reading a stored endog_grid when
    model_config["upper_envelope"]["skip_endog_grid_storage"] is True. Expects
    continuous_states_info = model_config["continuous_states_info"].

    "dj_wealth_grid" is None when assets_begin_of_period is state-choice-specific
    (see check_model_config.py) -- there is no single shared array to broadcast in
    that case. Callers of this function only reach it for the additional-
    continuous-state (n-D regular) interpolation path, which
    process_continuous_grid_functions already forbids combining with a
    state-specific assets_begin_of_period; the simple 1d case computes each
    state-choice's own grid on demand instead (see interp_interfaces.py /
    simulation_interp.py), ignoring this value entirely. So a zero placeholder here
    is always safe -- it is never actually read when it would be wrong.

    """
    if continuous_states_info["dj_wealth_grid"] is None:
        return jnp.zeros(shape)
    return jnp.broadcast_to(continuous_states_info["dj_wealth_grid"], shape)
