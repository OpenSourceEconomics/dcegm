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
    that case, so a zero placeholder of the right shape is returned instead.

    That placeholder is never read: every reader that receives it recomputes the
    state-choice's own grid on demand, both on the simple 1d path
    (_dj_wealth_grid_for_state_choice) and on the additional-continuous-state (n-D
    regular) path (_interp_policy_and_value_multidim_dj_for_state_choice in
    interp_interfaces.py, interpnd_policy_and_value_function in
    simulation_interp.py). Only the array's *shape* matters here, to keep vmap's
    in_axes=None contract.

    An earlier version of this docstring justified the placeholder differently --
    that process_continuous_grid_functions forbade combining a state-specific
    assets_begin_of_period with additional continuous states, so the n-D readers
    could never see it. That restriction was later lifted for the
    skip_endog_grid_storage case, which silently invalidated the justification and
    left those readers interpolating against zeros until the n-D recomputation
    above was added.

    """
    if continuous_states_info["dj_wealth_grid"] is None:
        return jnp.zeros(shape)
    return jnp.broadcast_to(continuous_states_info["dj_wealth_grid"], shape)
