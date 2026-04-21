"""N-dimensional regular-grid interpolation for policy/value.

Assumptions:
- Shared 1D wealth grid across all regular-grid combinations.
- Child-state policy/value grids are flattened in regular dimensions:
  ``(n_child_state_choices, n_continuous_combinations, n_wealth)``.
- Child-state interpolation points are provided as
  ``continuous_state_child_states[name]`` with shape
  ``(n_child_state_choices, n_continuous_combinations)``.

"""

from typing import Any, Callable, Dict

import jax.numpy as jnp
from jax import vmap

from dcegm.interpolation.interp1d import (
    get_index_high_and_low,
    linear_interpolation_formula,
)


def interpnd_policy_for_child_states_on_regular_grids(
    additional_continuous_state_grids: Dict[str, jnp.ndarray],
    wealth_grid: jnp.ndarray,
    policy_grid_child_states: jnp.ndarray,
    value_grid_child_states: jnp.ndarray,
    continuous_state_child_states: Dict[str, jnp.ndarray],
    wealth_child_states: jnp.ndarray,
    state_choice_child_states: Dict[str, Any],
    compute_utility: Callable,
    params: Dict[str, Any],
    discount_factor: float,
) -> jnp.ndarray:
    """Interpolate policy, using value-based overwrite logic.

    Returns shape
    ``(n_child_state_choices, n_continuous_combinations, n_wealth, n_quad_points)``.

    """
    policy, _ = interpnd_policy_and_value_for_child_states_on_regular_grids(
        additional_continuous_state_grids=additional_continuous_state_grids,
        wealth_grid=wealth_grid,
        policy_grid_child_states=policy_grid_child_states,
        value_grid_child_states=value_grid_child_states,
        continuous_state_child_states=continuous_state_child_states,
        wealth_child_states=wealth_child_states,
        state_choice_child_states=state_choice_child_states,
        compute_utility=compute_utility,
        params=params,
        discount_factor=discount_factor,
    )
    return policy


def interpnd_policy_and_value_for_child_states_on_regular_grids(
    additional_continuous_state_grids: Dict[str, jnp.ndarray],
    wealth_grid: jnp.ndarray,
    policy_grid_child_states: jnp.ndarray,
    value_grid_child_states: jnp.ndarray,
    continuous_state_child_states: Dict[str, jnp.ndarray],
    wealth_child_states: jnp.ndarray,
    state_choice_child_states: Dict[str, Any],
    compute_utility: Callable,
    params: Dict[str, Any],
    discount_factor: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate policy/value and apply consume-all overwrite.

    If consume-all value dominates interpolated value at a point, overwrite policy with
    consume-all policy (=wealth point).

    """
    objs = _precompute_interp_objects(
        additional_continuous_state_grids=additional_continuous_state_grids,
        continuous_state_child_states=continuous_state_child_states,
        wealth_grid=wealth_grid,
        wealth_child_states=wealth_child_states,
    )

    def _interp_one_child_state(
        policy_grid_one_child,
        value_grid_one_child,
        regular_low_idxs_one_child,
        regular_high_idxs_one_child,
        regular_low_weights_one_child,
        regular_high_weights_one_child,
        wealth_points_one_child,
        wealth_low_idxs_one_child,
        wealth_high_idxs_one_child,
    ):
        return vmap(
            _interp_policy_and_value_one_comb,
            in_axes=(
                None,
                None,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                None,
                None,
                None,
            ),
        )(
            policy_grid_one_child,
            value_grid_one_child,
            regular_low_idxs_one_child,
            regular_high_idxs_one_child,
            regular_low_weights_one_child,
            regular_high_weights_one_child,
            wealth_points_one_child,
            wealth_low_idxs_one_child,
            wealth_high_idxs_one_child,
            objs["strides"],
            objs["corner_table"],
            wealth_grid,
        )

    policy_interp, value_interp = vmap(
        _interp_one_child_state,
        in_axes=(0, 0, 1, 1, 1, 1, 0, 0, 0),
    )(
        policy_grid_child_states,
        value_grid_child_states,
        objs["regular_low_idxs"],
        objs["regular_high_idxs"],
        objs["regular_low_weights"],
        objs["regular_high_weights"],
        wealth_child_states,
        objs["wealth_low_idxs"],
        objs["wealth_high_idxs"],
    )

    # We need to interpolate the expected value at zero savings, because we only know it for the regular
    # grid corners
    expected_value_zero_savings = _interp_regular_only_all(
        values_over_regular_grid_child_states=value_grid_child_states[..., 0],
        regular_low_idxs=objs["regular_low_idxs"],
        regular_high_idxs=objs["regular_high_idxs"],
        regular_low_weights=objs["regular_low_weights"],
        regular_high_weights=objs["regular_high_weights"],
        strides=objs["strides"],
        corner_table=objs["corner_table"],
    )

    consume_all_value = _compute_consume_all_value(
        expected_value_zero_savings=expected_value_zero_savings,
        wealth_child_states=wealth_child_states,
        state_choice_child_states=state_choice_child_states,
        continuous_state_child_states=continuous_state_child_states,
        compute_utility=compute_utility,
        params=params,
        discount_factor=discount_factor,
    )

    overwrite_mask = consume_all_value > value_interp
    policy_final = jnp.where(overwrite_mask, wealth_child_states, policy_interp)
    value_final = jnp.where(overwrite_mask, consume_all_value, value_interp)
    return policy_final, value_final


def interpnd_value_for_child_states_on_regular_grids(
    additional_continuous_state_grids: Dict[str, jnp.ndarray],
    wealth_grid: jnp.ndarray,
    value_grid_child_states: jnp.ndarray,
    continuous_state_child_states: Dict[str, jnp.ndarray],
    wealth_child_states: jnp.ndarray,
    state_choice_child_states: Dict[str, Any],
    compute_utility: Callable,
    params: Dict[str, Any],
    discount_factor: float,
) -> jnp.ndarray:
    """Interpolate value and apply consume-all overwrite.

    Returns shape
    ``(n_child_state_choices, n_continuous_combinations, n_wealth, n_quad_points)``.

    """
    objs = _precompute_interp_objects(
        additional_continuous_state_grids=additional_continuous_state_grids,
        continuous_state_child_states=continuous_state_child_states,
        wealth_grid=wealth_grid,
        wealth_child_states=wealth_child_states,
    )

    def _interp_one_child_state(
        value_grid_one_child,
        regular_low_idxs_one_child,
        regular_high_idxs_one_child,
        regular_low_weights_one_child,
        regular_high_weights_one_child,
        wealth_points_one_child,
        wealth_low_idxs_one_child,
        wealth_high_idxs_one_child,
    ):
        def _interp_one_comb(
            regular_low_idxs_one_comb,
            regular_high_idxs_one_comb,
            regular_low_weights_one_comb,
            regular_high_weights_one_comb,
            wealth_points_one_comb,
            wealth_low_idxs_one_comb,
            wealth_high_idxs_one_comb,
        ):
            corner_linear_idxs, corner_weights = _corner_linear_indices_and_weights(
                regular_low_idxs_one_comb=regular_low_idxs_one_comb,
                regular_high_idxs_one_comb=regular_high_idxs_one_comb,
                regular_low_weights_one_comb=regular_low_weights_one_comb,
                regular_high_weights_one_comb=regular_high_weights_one_comb,
                strides=objs["strides"],
                corner_table=objs["corner_table"],
            )
            return _interp_single_grid_one_comb(
                grid_one_child=value_grid_one_child,
                corner_linear_idxs=corner_linear_idxs,
                corner_weights=corner_weights,
                wealth_points_one_comb=wealth_points_one_comb,
                wealth_low_idxs_one_comb=wealth_low_idxs_one_comb,
                wealth_high_idxs_one_comb=wealth_high_idxs_one_comb,
                wealth_grid=wealth_grid,
            )

        return vmap(
            _interp_one_comb,
            in_axes=(1, 1, 1, 1, 0, 0, 0),
        )(
            regular_low_idxs_one_child,
            regular_high_idxs_one_child,
            regular_low_weights_one_child,
            regular_high_weights_one_child,
            wealth_points_one_child,
            wealth_low_idxs_one_child,
            wealth_high_idxs_one_child,
        )

    value_interp = vmap(
        _interp_one_child_state,
        in_axes=(0, 1, 1, 1, 1, 0, 0, 0),
    )(
        value_grid_child_states,
        objs["regular_low_idxs"],
        objs["regular_high_idxs"],
        objs["regular_low_weights"],
        objs["regular_high_weights"],
        wealth_child_states,
        objs["wealth_low_idxs"],
        objs["wealth_high_idxs"],
    )

    expected_value_zero_savings = _interp_regular_only_all(
        values_over_regular_grid_child_states=value_grid_child_states[..., 0],
        regular_low_idxs=objs["regular_low_idxs"],
        regular_high_idxs=objs["regular_high_idxs"],
        regular_low_weights=objs["regular_low_weights"],
        regular_high_weights=objs["regular_high_weights"],
        strides=objs["strides"],
        corner_table=objs["corner_table"],
    )

    consume_all_value = _compute_consume_all_value(
        expected_value_zero_savings=expected_value_zero_savings,
        wealth_child_states=wealth_child_states,
        state_choice_child_states=state_choice_child_states,
        continuous_state_child_states=continuous_state_child_states,
        compute_utility=compute_utility,
        params=params,
        discount_factor=discount_factor,
    )

    return jnp.asarray(
        jnp.where(consume_all_value > value_interp, consume_all_value, value_interp)
    )


def _compute_consume_all_value(
    expected_value_zero_savings: jnp.ndarray,
    wealth_child_states: jnp.ndarray,
    state_choice_child_states: Dict[str, Any],
    continuous_state_child_states: Dict[str, jnp.ndarray],
    compute_utility: Callable,
    params: Dict[str, Any],
    discount_factor: float,
) -> jnp.ndarray:

    def _utility_at_point(
        consumption_point: jnp.ndarray,
        state_choice_point: Dict[str, jnp.ndarray],
        continuous_state_point: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        out = compute_utility(
            consumption=consumption_point,
            params=params,
            **state_choice_point,
            **continuous_state_point,
        )
        return out

    consume_all_utility = vmap(
        vmap(
            vmap(
                vmap(
                    _utility_at_point,
                    in_axes=(0, None, None),
                ),
                in_axes=(0, None, None),
            ),
            in_axes=(0, None, 0),
        ),
        in_axes=(0, 0, 0),
    )(
        wealth_child_states,
        state_choice_child_states,
        continuous_state_child_states,
    )

    expected_value_zero_savings = expected_value_zero_savings[:, :, None, None]
    return consume_all_utility + discount_factor * expected_value_zero_savings


def _interp_policy_and_value_one_comb(
    policy_grid_one_child: jnp.ndarray,
    value_grid_one_child: jnp.ndarray,
    regular_low_idxs_one_comb: jnp.ndarray,
    regular_high_idxs_one_comb: jnp.ndarray,
    regular_low_weights_one_comb: jnp.ndarray,
    regular_high_weights_one_comb: jnp.ndarray,
    wealth_points_one_comb: jnp.ndarray,
    wealth_low_idxs_one_comb: jnp.ndarray,
    wealth_high_idxs_one_comb: jnp.ndarray,
    strides: jnp.ndarray,
    corner_table: jnp.ndarray,
    wealth_grid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    corner_linear_idxs, corner_weights = _corner_linear_indices_and_weights(
        regular_low_idxs_one_comb=regular_low_idxs_one_comb,
        regular_high_idxs_one_comb=regular_high_idxs_one_comb,
        regular_low_weights_one_comb=regular_low_weights_one_comb,
        regular_high_weights_one_comb=regular_high_weights_one_comb,
        strides=strides,
        corner_table=corner_table,
    )

    policy_interp = _interp_single_grid_one_comb(
        grid_one_child=policy_grid_one_child,
        corner_linear_idxs=corner_linear_idxs,
        corner_weights=corner_weights,
        wealth_points_one_comb=wealth_points_one_comb,
        wealth_low_idxs_one_comb=wealth_low_idxs_one_comb,
        wealth_high_idxs_one_comb=wealth_high_idxs_one_comb,
        wealth_grid=wealth_grid,
    )
    value_interp = _interp_single_grid_one_comb(
        grid_one_child=value_grid_one_child,
        corner_linear_idxs=corner_linear_idxs,
        corner_weights=corner_weights,
        wealth_points_one_comb=wealth_points_one_comb,
        wealth_low_idxs_one_comb=wealth_low_idxs_one_comb,
        wealth_high_idxs_one_comb=wealth_high_idxs_one_comb,
        wealth_grid=wealth_grid,
    )
    return policy_interp, value_interp


def _interp_single_grid_one_comb(
    grid_one_child: jnp.ndarray,
    corner_linear_idxs: jnp.ndarray,
    corner_weights: jnp.ndarray,
    wealth_points_one_comb: jnp.ndarray,
    wealth_low_idxs_one_comb: jnp.ndarray,
    wealth_high_idxs_one_comb: jnp.ndarray,
    wealth_grid: jnp.ndarray,
) -> jnp.ndarray:
    corner_rows = grid_one_child[corner_linear_idxs]
    corner_values = vmap(
        _interp_wealth_for_corner,
        in_axes=(0, None, None, None, None),
    )(
        corner_rows,
        wealth_points_one_comb,
        wealth_low_idxs_one_comb,
        wealth_high_idxs_one_comb,
        wealth_grid,
    )
    return jnp.sum(corner_weights[:, None, None] * corner_values, axis=0)


def _interp_wealth_for_corner(
    grid_row: jnp.ndarray,
    wealth_points_one_comb: jnp.ndarray,
    wealth_low_idxs_one_comb: jnp.ndarray,
    wealth_high_idxs_one_comb: jnp.ndarray,
    wealth_grid: jnp.ndarray,
) -> jnp.ndarray:
    high = _take_last_axis(grid_row, wealth_high_idxs_one_comb)
    low = _take_last_axis(grid_row, wealth_low_idxs_one_comb)
    out = linear_interpolation_formula(
        y_high=high,
        y_low=low,
        x_high=wealth_grid[wealth_high_idxs_one_comb],
        x_low=wealth_grid[wealth_low_idxs_one_comb],
        x_new=wealth_points_one_comb,
    )
    return jnp.asarray(out)


def _interp_regular_only_all(
    values_over_regular_grid_child_states: jnp.ndarray,
    regular_low_idxs: jnp.ndarray,
    regular_high_idxs: jnp.ndarray,
    regular_low_weights: jnp.ndarray,
    regular_high_weights: jnp.ndarray,
    strides: jnp.ndarray,
    corner_table: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate values over regular dimensions only."""

    def _interp_one_child_state(
        values_over_regular_grid_one_child,
        regular_low_idxs_one_child,
        regular_high_idxs_one_child,
        regular_low_weights_one_child,
        regular_high_weights_one_child,
    ):
        return vmap(
            _interp_regular_only,
            in_axes=(None, 1, 1, 1, 1, None, None),
        )(
            values_over_regular_grid_one_child,
            regular_low_idxs_one_child,
            regular_high_idxs_one_child,
            regular_low_weights_one_child,
            regular_high_weights_one_child,
            strides,
            corner_table,
        )

    return vmap(
        _interp_one_child_state,
        in_axes=(0, 1, 1, 1, 1),
    )(
        values_over_regular_grid_child_states,
        regular_low_idxs,
        regular_high_idxs,
        regular_low_weights,
        regular_high_weights,
    )


def _corner_linear_indices_and_weights(
    regular_low_idxs_one_comb: jnp.ndarray,
    regular_high_idxs_one_comb: jnp.ndarray,
    regular_low_weights_one_comb: jnp.ndarray,
    regular_high_weights_one_comb: jnp.ndarray,
    strides: jnp.ndarray,
    corner_table: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    choose_high = corner_table.astype(bool)
    selected_idxs = jnp.where(
        choose_high,
        regular_high_idxs_one_comb[None, :],
        regular_low_idxs_one_comb[None, :],
    )
    selected_weights = jnp.where(
        choose_high,
        regular_high_weights_one_comb[None, :],
        regular_low_weights_one_comb[None, :],
    )
    corner_linear_idxs = jnp.sum(selected_idxs * strides[None, :], axis=1)
    corner_weights = jnp.prod(selected_weights, axis=1)
    return corner_linear_idxs, corner_weights


def _interp_regular_only(
    values_over_regular_grid: jnp.ndarray,
    regular_low_idxs_one_comb: jnp.ndarray,
    regular_high_idxs_one_comb: jnp.ndarray,
    regular_low_weights_one_comb: jnp.ndarray,
    regular_high_weights_one_comb: jnp.ndarray,
    strides: jnp.ndarray,
    corner_table: jnp.ndarray,
) -> jnp.ndarray:
    corner_linear_idxs, corner_weights = _corner_linear_indices_and_weights(
        regular_low_idxs_one_comb=regular_low_idxs_one_comb,
        regular_high_idxs_one_comb=regular_high_idxs_one_comb,
        regular_low_weights_one_comb=regular_low_weights_one_comb,
        regular_high_weights_one_comb=regular_high_weights_one_comb,
        strides=strides,
        corner_table=corner_table,
    )
    corner_vals = values_over_regular_grid[corner_linear_idxs]
    return jnp.sum(corner_weights * corner_vals)


def _precompute_interp_objects(
    additional_continuous_state_grids: Dict[str, jnp.ndarray],
    continuous_state_child_states: Dict[str, jnp.ndarray],
    wealth_grid: jnp.ndarray,
    wealth_child_states: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Precompute reusable interpolation objects for policy/value paths."""
    state_names = list(additional_continuous_state_grids.keys())
    regular_shape = [
        int(additional_continuous_state_grids[name].shape[0]) for name in state_names
    ]
    strides = jnp.asarray(_compute_row_major_strides(regular_shape), dtype=jnp.int32)
    regular_low_idxs, regular_high_idxs, regular_low_weights, regular_high_weights = (
        _precompute_regular_indices_and_weights(
            additional_continuous_state_grids=additional_continuous_state_grids,
            continuous_state_child_states=continuous_state_child_states,
            state_names=state_names,
        )
    )
    wealth_high_idxs, wealth_low_idxs = get_index_high_and_low(
        wealth_grid, wealth_child_states
    )
    corner_table = _corner_table(len(state_names))
    return {
        "strides": strides,
        "regular_low_idxs": regular_low_idxs,
        "regular_high_idxs": regular_high_idxs,
        "regular_low_weights": regular_low_weights,
        "regular_high_weights": regular_high_weights,
        "wealth_low_idxs": wealth_low_idxs,
        "wealth_high_idxs": wealth_high_idxs,
        "corner_table": corner_table,
    }


def _precompute_regular_indices_and_weights(
    additional_continuous_state_grids: Dict[str, jnp.ndarray],
    continuous_state_child_states: Dict[str, jnp.ndarray],
    state_names,
):
    """Precompute low/high idx and weights for all regular child points.

    Returns arrays of shape
    ``(n_dims, n_child_state_choices, n_continuous_combinations)``.

    """
    low_idxs = []
    high_idxs = []
    low_weights = []
    high_weights = []

    for name in state_names:
        grid_1d = additional_continuous_state_grids[name]
        points = continuous_state_child_states[name]
        high_idx, low_idx = get_index_high_and_low(grid_1d, points)
        x_low = grid_1d[low_idx]
        x_high = grid_1d[high_idx]
        high_w = (points - x_low) / (x_high - x_low)
        low_w = 1.0 - high_w
        low_idxs.append(low_idx)
        high_idxs.append(high_idx)
        low_weights.append(low_w)
        high_weights.append(high_w)

    return (
        jnp.stack(low_idxs, axis=0),
        jnp.stack(high_idxs, axis=0),
        jnp.stack(low_weights, axis=0),
        jnp.stack(high_weights, axis=0),
    )


def _take_last_axis(arr, idx):
    """Take along last axis with batched indices."""
    if arr.ndim != idx.ndim + 1:
        n_missing = idx.ndim - (arr.ndim - 1)
        arr = arr.reshape(arr.shape[:-1] + (1,) * n_missing + (arr.shape[-1],))
        arr = jnp.broadcast_to(arr, idx.shape + (arr.shape[-1],))
    return jnp.take_along_axis(arr, idx[..., None], axis=-1)[..., 0]


def _compute_row_major_strides(shape):
    strides = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = running
        running *= shape[i]
    return strides


def _corner_table(n_dims: int) -> jnp.ndarray:
    """Return binary corner table of shape ``(2**n_dims, n_dims)``.

    Pseudo-code equivalent:

    for corner in range(2**n_dims):
        for dim in range(n_dims):
            table[corner, dim] = (corner >> dim) & 1

    """
    corners = jnp.arange(2**n_dims, dtype=jnp.int32)
    shifts = jnp.arange(n_dims, dtype=jnp.int32)
    return ((corners[:, None] >> shifts[None, :]) & 1).astype(jnp.int32)
