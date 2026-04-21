"""N-dimensional regular-grid interpolation for policy (minimal).

Assumptions for this module:
- Wealth grid is shared across all regular-grid combinations.
- Policy grid for child states has shape
  ``(n_child_state_choices, n_continuous_combinations, n_wealth)``.
- Continuous child states are passed as a dictionary with values of shape
  ``(n_child_state_choices, n_continuous_combinations)``.
- Additional continuous-state grids are passed as a dictionary with one 1D grid
  per continuous state name.

"""

from typing import Dict

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
    continuous_state_child_states: Dict[str, jnp.ndarray],
    wealth_child_states: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate policy for all child states and all interpolation points.

    Args:
        additional_continuous_state_grids: Dict[name -> 1D grid].
        wealth_grid: Shared 1D wealth grid, shape ``(n_wealth,)``.
        policy_grid_child_states: Policy in child states,
            shape ``(n_child_state_choices, n_continuous_combinations, n_wealth)``.
        continuous_state_child_states: Dict[name -> values], with shape
            ``(n_child_state_choices, n_continuous_combinations)`` per value.
        wealth_child_states: Wealth interpolation points,
            shape ``(n_child_state_choices, n_continuous_combinations, n_wealth, n_quad_points)``.

    Returns:
        Interpolated policy with shape
        ``(n_child_state_choices, n_continuous_combinations, n_wealth, n_quad_points)``.

    """
    # `state_names` defines the interpolation axes ordering.
    # The same ordering is used for:
    # - the meshgrid flattening in preprocessing,
    # - stride construction below,
    # - corner index aggregation.
    state_names = list(additional_continuous_state_grids.keys())
    # Number of grid points per regular axis, shape: (n_dims,).
    # Example with exp_green(5), exp_red(4): regular_shape = [5, 4].
    regular_shape = [
        int(additional_continuous_state_grids[name].shape[0]) for name in state_names
    ]
    # Row-major strides map an N-D index to the flattened
    # `n_continuous_combinations` index.
    strides = jnp.asarray(_compute_row_major_strides(regular_shape), dtype=jnp.int32)

    # -------------------------------------------------------------------------
    # Precompute index brackets and weights for ALL regular child points.
    # Shapes (per returned array):
    #   (n_dims, n_child_state_choices, n_continuous_combinations)
    # -------------------------------------------------------------------------
    regular_low_idxs, regular_high_idxs, regular_low_weights, regular_high_weights = (
        _precompute_regular_indices_and_weights(
            additional_continuous_state_grids=additional_continuous_state_grids,
            continuous_state_child_states=continuous_state_child_states,
            state_names=state_names,
        )
    )

    # -------------------------------------------------------------------------
    # Precompute wealth brackets for ALL wealth child points.
    # Shapes:
    #   (n_child_state_choices, n_continuous_combinations, n_wealth, n_quad_points)
    # -------------------------------------------------------------------------
    wealth_high_idxs, wealth_low_idxs = get_index_high_and_low(
        wealth_grid,
        wealth_child_states,
    )

    corner_table = _corner_table(len(state_names))

    def _interp_one_child_state(
        policy_grid_one_child,
        regular_low_idxs_one_child,
        regular_high_idxs_one_child,
        regular_low_weights_one_child,
        regular_high_weights_one_child,
        wealth_points_one_child,
        wealth_low_idxs_one_child,
        wealth_high_idxs_one_child,
    ):
        # policy_grid_one_child: (n_continuous_combinations, n_wealth)
        # *_one_child regular arrays: (n_dims, n_continuous_combinations)
        # wealth_points_one_child / wealth idxs one child:
        #   (n_continuous_combinations, n_wealth, n_quad_points)
        return vmap(
            _interp_one_continuous_combination,
            in_axes=(
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
            regular_low_idxs_one_child,
            regular_high_idxs_one_child,
            regular_low_weights_one_child,
            regular_high_weights_one_child,
            wealth_points_one_child,
            wealth_low_idxs_one_child,
            wealth_high_idxs_one_child,
            strides,
            corner_table,
            wealth_grid,
        )

    # -------------------------------------------------------------------------
    # vmap over n_child_state_choices (outer-most axis)
    # -------------------------------------------------------------------------
    return vmap(
        _interp_one_child_state,
        in_axes=(0, 1, 1, 1, 1, 0, 0, 0),
    )(
        policy_grid_child_states,
        regular_low_idxs,
        regular_high_idxs,
        regular_low_weights,
        regular_high_weights,
        wealth_child_states,
        wealth_low_idxs,
        wealth_high_idxs,
    )


def _interp_one_continuous_combination(
    policy_grid_one_child: jnp.ndarray,
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
) -> jnp.ndarray:
    """Interpolate one (child choice, continuous combination) block.

    Returns shape ``(n_wealth, n_quad_points)``.

    """
    # `n_dims` is the number of additional continuous-state axes.
    # For K additional continuous states, n_dims = K and
    # n_corners = 2**K.

    # corner_table: (n_corners, n_dims), values in {0,1}
    # 0 means choose low index on that axis, 1 means choose high index.
    choose_high = corner_table.astype(bool)

    # Vectorized corner index selection.
    # selected_* shapes: (n_corners, n_dims)
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

    # Corner linear indices and tensor-product weights.
    # corner_linear_idxs: (n_corners,)
    # corner_weights: (n_corners,)
    corner_linear_idxs = jnp.sum(selected_idxs * strides[None, :], axis=1)
    corner_weights = jnp.prod(selected_weights, axis=1)

    # Gather all corner rows in one shot: (n_corners, n_wealth).
    # Each row corresponds to one regular-grid corner.
    policy_rows = policy_grid_one_child[corner_linear_idxs]

    # Interpolate in wealth for each corner.
    def _interp_wealth_for_corner(policy_row):
        policy_high = _take_last_axis(policy_row, wealth_high_idxs_one_comb)
        policy_low = _take_last_axis(policy_row, wealth_low_idxs_one_comb)
        return linear_interpolation_formula(
            y_high=policy_high,
            y_low=policy_low,
            x_high=wealth_grid[wealth_high_idxs_one_comb],
            x_low=wealth_grid[wealth_low_idxs_one_comb],
            x_new=wealth_points_one_comb,
        )

    # policy_corner_values: (n_corners, n_wealth, n_quad_points)
    policy_corner_values = vmap(_interp_wealth_for_corner, in_axes=0)(policy_rows)

    # Aggregate corners.
    return jnp.sum(
        corner_weights[:, None, None] * policy_corner_values,
        axis=0,
    )


def _precompute_regular_indices_and_weights(
    additional_continuous_state_grids: Dict[str, jnp.ndarray],
    continuous_state_child_states: Dict[str, jnp.ndarray],
    state_names,
):
    """Precompute low/high idx and weights for all regular child points.

    Returns four arrays of shape
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

    `n_dims` is the number of additional continuous-state axes.
    Example for n_dims=2, returned rows are:
    [0,0], [1,0], [0,1], [1,1].

    Pseudo-code equivalent (more readable but less vectorized):

    for corner in range(2**n_dims):
        for dim in range(n_dims):
            table[corner, dim] = (corner >> dim) & 1

    Here, value 0 means "use low index" and 1 means "use high index" on that
    regular axis.

    """
    corners = jnp.arange(2**n_dims, dtype=jnp.int32)
    shifts = jnp.arange(n_dims, dtype=jnp.int32)
    return ((corners[:, None] >> shifts[None, :]) & 1).astype(jnp.int32)


if __name__ == "__main__":
    # Minimal executable shape example.
    # Regular grids (2 dimensions): 3 x 2 => 6 combinations.
    additional_cont_grids = {
        "exp_green": jnp.array([0.0, 0.5, 1.0]),
        "exp_red": jnp.array([0.0, 1.0]),
    }
    m_grid = jnp.array([0.0, 5.0, 10.0, 20.0])

    n_child_state_choices = 2
    n_cont = 6
    n_wealth = 2
    n_quad = 3

    policy_grid_child = jnp.arange(
        n_child_state_choices * n_cont * m_grid.shape[0],
        dtype=jnp.float32,
    ).reshape(n_child_state_choices, n_cont, m_grid.shape[0])

    continuous_child = {
        "exp_green": jnp.array(
            [[0.1, 0.4, 0.7, 0.2, 0.5, 0.8], [0.1, 0.4, 0.7, 0.2, 0.5, 0.8]]
        ),
        "exp_red": jnp.array(
            [[0.2, 0.2, 0.2, 0.7, 0.7, 0.7], [0.2, 0.2, 0.2, 0.7, 0.7, 0.7]]
        ),
    }
    wealth_child = jnp.array(
        [
            [
                [[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]],
            ]
            * n_cont,
            [
                [[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]],
            ]
            * n_cont,
        ]
    )

    out = interpnd_policy_for_child_states_on_regular_grids(
        additional_continuous_state_grids=additional_cont_grids,
        wealth_grid=m_grid,
        policy_grid_child_states=policy_grid_child,
        continuous_state_child_states=continuous_child,
        wealth_child_states=wealth_child,
    )

    print("policy_grid_child_states shape:", tuple(policy_grid_child.shape))
    print(
        "continuous_state_child_states shape (per key):",
        {k: tuple(v.shape) for k, v in continuous_child.items()},
    )
    print("wealth_child_states shape:", tuple(wealth_child.shape))
    print("output shape:", tuple(out.shape))
