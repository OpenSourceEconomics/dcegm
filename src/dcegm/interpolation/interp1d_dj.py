from typing import Callable, Dict, Tuple

import jax.numpy as jnp

from dcegm.interpolation.interp1d import (
    get_index_high_and_low,
    linear_interpolation_formula,
)


def interp1d_policy_and_value_on_wealth_dj(
    wealth: float | jnp.ndarray,
    wealth_grid: jnp.ndarray,
    policy_grid: jnp.ndarray,
    value_grid: jnp.ndarray,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    params: Dict[str, float],
    discount_factor: float,
) -> Tuple[jnp.ndarray | float, jnp.ndarray | float]:
    """1D interpolation for DJ with consume-all overwrite for policy and value."""
    ind_high, ind_low = get_index_high_and_low(x=wealth_grid, x_new=wealth)

    policy_interp = linear_interpolation_formula(
        y_high=policy_grid[ind_high],
        y_low=policy_grid[ind_low],
        x_high=wealth_grid[ind_high],
        x_low=wealth_grid[ind_low],
        x_new=wealth,
    )
    value_interp_on_grid = linear_interpolation_formula(
        y_high=value_grid[ind_high],
        y_low=value_grid[ind_low],
        x_high=wealth_grid[ind_high],
        x_low=wealth_grid[ind_low],
        x_new=wealth,
    )

    consume_all_value = _consume_all_value(
        wealth=wealth,
        value_at_zero_wealth=value_grid[0],
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
        discount_factor=discount_factor,
    )
    overwrite_mask = consume_all_value > value_interp_on_grid
    policy = jnp.where(overwrite_mask, wealth, policy_interp)
    value = jnp.where(overwrite_mask, consume_all_value, value_interp_on_grid)
    return policy, value


def interp1d_value_on_wealth_dj(
    wealth: float | jnp.ndarray,
    wealth_grid: jnp.ndarray,
    value_grid: jnp.ndarray,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    params: Dict[str, float],
    discount_factor: float,
) -> jnp.ndarray | float:
    """1D value interpolation for DJ with consume-all overwrite."""
    _, value = interp1d_policy_and_value_on_wealth_dj(
        wealth=wealth,
        wealth_grid=wealth_grid,
        policy_grid=wealth_grid,
        value_grid=value_grid,
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
        discount_factor=discount_factor,
    )
    return value


def _consume_all_value(
    wealth: float | jnp.ndarray,
    value_at_zero_wealth: float | jnp.ndarray,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    params: Dict[str, float],
    discount_factor: float,
) -> jnp.ndarray:
    util = compute_utility(consumption=wealth, params=params, **state_choice_vec)
    if isinstance(util, tuple):
        util = util[0]
    return jnp.asarray(util) + discount_factor * value_at_zero_wealth
