from typing import List, Tuple

import jax
import jax.numpy as jnp

from dcegm.interpolation.interp1d import get_index_high_and_low


def _regular_indices_and_weights(
    regular_grids: List[jnp.ndarray], regular_point: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    For each regular axis i:
      - find low/high index (i_lo, i_hi)
      - compute 1D weight t_i in [0,1]
    Returns:
      idx_lo: (R,), idx_hi: (R,), t: (R,)
    """

    def one_axis(g, x):
        hi, lo = get_index_high_and_low(g, x)
        # Guard against zero division (degenerate cell)
        denom = jnp.maximum(g[hi] - g[lo], jnp.finfo(g.dtype).eps)
        t = (x - g[lo]) / denom
        return lo, hi, t

    lo, hi, t = jax.vmap(one_axis, in_axes=(0, 0))(
        jnp.array(regular_grids, dtype=object), regular_point
    )
    # The dtype=object trick is not JIT-able; instead, pass as tuple and vmapping won't work.
    # We'll implement with a Python loop since R is static, which is JIT-safe.


def _regular_indices_and_weights_static(
    regular_grids: List[jnp.ndarray], regular_point: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    R = len(regular_grids)
    idx_lo = []
    idx_hi = []
    ts = []
    for i in range(R):
        hi, lo = get_index_high_and_low(regular_grids[i], regular_point[i])
        g = regular_grids[i]
        denom = jnp.maximum(g[hi] - g[lo], jnp.finfo(g.dtype).eps)
        t = (regular_point[i] - g[lo]) / denom
        idx_lo.append(lo)
        idx_hi.append(hi)
        ts.append(t)
    return jnp.array(idx_lo), jnp.array(idx_hi), jnp.array(ts)


def _enumerate_corners(R: int) -> jnp.ndarray:
    """Returns a (2**R, R) array of {0,1} bits.

    Row k gives the low/high selector per axis for corner k.

    """
    C = 1 << R
    ks = jnp.arange(C, dtype=jnp.uint32)[:, None]  # (C,1)
    axes = jnp.arange(R, dtype=jnp.uint32)[None, :]  # (1,R)
    return (ks >> axes) & 1  # (C,R)


def _flat_strides(dims: Tuple[int, ...]) -> jnp.ndarray:
    """Given dims = (n0,...,n{R-1}), return strides for row-major flattening.

    stride[i] = prod_{j<i} n_j, with stride[0]=1.

    """
    if not dims:
        return jnp.array([], dtype=jnp.int32)
    return jnp.concatenate(
        [
            jnp.array([1], dtype=jnp.int32),
            jnp.cumprod(jnp.array(dims[:-1], dtype=jnp.int32)),
        ]
    )


def interpNd_one_irregular(
    regular_grids: List[jnp.ndarray],  # list of R arrays (n_i,)
    irregular_grid: jnp.ndarray,  # shape: (n0,...,n{R-1}, nW)
    values: jnp.ndarray,  # shape: (n0,...,n{R-1}, nW)
    regular_point: jnp.ndarray,  # shape: (R,)
    irregular_point: float,  # scalar
) -> jnp.ndarray:
    """N-D linear interpolation with exactly one irregular axis (the last one).

    - Per-corner: 1D linear interpolation along the irregular axis using the local (varying) grid
    - Then: multilinear blend across regular axes with tensor-product weights

    Returns scalar (same dtype as `values`).

    """
    R = len(regular_grids)
    dims = tuple(g.shape[0] for g in regular_grids)
    nW = irregular_grid.shape[-1]

    # 1) indices and weights for regular axes
    idx_lo, idx_hi, t = _regular_indices_and_weights_static(
        regular_grids, regular_point
    )  # (R,), (R,), (R,)

    # 2) enumerate corners and build corner-specific regular indices
    sel = _enumerate_corners(R).astype(jnp.int32)  # (C,R), 0=lo,1=hi
    idx_lo_b = jnp.broadcast_to(idx_lo, sel.shape)  # (C,R)
    idx_hi_b = jnp.broadcast_to(idx_hi, sel.shape)  # (C,R)
    corner_idx = jnp.where(sel == 0, idx_lo_b, idx_hi_b)  # (C,R)

    # 3) compute tensor-product weights across regular axes
    t_b = jnp.broadcast_to(t, sel.shape)  # (C,R)
    w_axes = jnp.where(sel == 0, 1.0 - t_b, t_b)  # (C,R)
    w_corners = jnp.prod(w_axes, axis=1)  # (C,)

    # 4) flatten gather of the corner rows (over regular block)
    strides = _flat_strides(dims)  # (R,)
    flat_idx = jnp.sum(corner_idx * strides, axis=1)  # (C,)

    irr_flat = irregular_grid.reshape((-1, nW))  # (Nreg, nW)
    val_flat = values.reshape((-1, nW))  # (Nreg, nW)

    irr_sel = irr_flat[flat_idx]  # (C, nW)
    val_sel = val_flat[flat_idx]  # (C, nW)

    # 5) per-corner 1D interpolation along irregular axis (vectorized)
    def interp1d_row(xrow, vrow, xnew):
        hi, lo = get_index_high_and_low(xrow, xnew)
        # Stack-indexing helpers
        lo_v = vrow[lo]
        hi_v = vrow[hi]
        lo_x = xrow[lo]
        hi_x = xrow[hi]
        denom = jnp.maximum(hi_x - lo_x, jnp.finfo(xrow.dtype).eps)
        s = (xnew - lo_x) / denom
        return lo_v + s * (hi_v - lo_v)

    z_corner = jax.vmap(interp1d_row, in_axes=(0, 0, None))(
        irr_sel, val_sel, irregular_point
    )  # (C,)

    # 6) final blend
    return jnp.sum(w_corners * z_corner)


def interpNd_policy(
    regular_grids, wealth_grid, policy_grid, regular_point, wealth_point
):
    return interpNd_one_irregular(
        regular_grids, wealth_grid, policy_grid, regular_point, wealth_point
    )


def interpNd_value_with_cc(
    regular_grids,
    wealth_grid,
    value_grid,
    regular_point,
    wealth_point,
    compute_utility,
    state_choice_vec,
    params,
    discount_factor,
):
    R = len(regular_grids)
    dims = tuple(g.shape[0] for g in regular_grids)
    nW = wealth_grid.shape[-1]

    # indices/weights and corners as above
    idx_lo, idx_hi, t = _regular_indices_and_weights_static(
        regular_grids, regular_point
    )
    sel = _enumerate_corners(R).astype(jnp.int32)
    idx_lo_b = jnp.broadcast_to(idx_lo, sel.shape)
    idx_hi_b = jnp.broadcast_to(idx_hi, sel.shape)
    corner_idx = jnp.where(sel == 0, idx_lo_b, idx_hi_b)

    strides = _flat_strides(dims)
    flat_idx = jnp.sum(corner_idx * strides, axis=1)  # (C,)

    wealth_min_unconstrained = wealth_grid[..., 1]  # (n0,...,n{R-1},)
    value_at_zero_wealth = value_grid[..., 0]  # (n0,...,n{R-1},)

    w_min_flat = wealth_min_unconstrained.reshape((-1,))
    v0_flat = value_at_zero_wealth.reshape((-1,))
    w_min_sel = w_min_flat[flat_idx]  # (C,)
    v0_sel = v0_flat[flat_idx]  # (C,)

    # closed-form corner value if constrained (consume all)
    v_cc = (
        compute_utility(
            consumption=wealth_point,
            params=params,
            continuous_state=regular_point,
            **state_choice_vec,
        )
        + discount_factor * v0_sel
    )  # (C,)

    # corner row slices of the value grid along irregular axis
    val_flat = value_grid.reshape((-1, nW))
    val_sel = val_flat[flat_idx]  # (C, nW)

    # replace whole row if constrained (same as your 2D left/right replacement, generalized)
    constrained = wealth_point <= w_min_sel  # (C,)
    # For constrained rows, the value at the *target* wealth is v_cc; emulate this
    # by performing 1D interpolation on a degenerate segment [wealth_point, wealth_point]
    # i.e., just overwrite the *interpolated* corner values after the 1D step.
    z_corner_unconstrained = jax.vmap(
        lambda xrow, vrow: _interp1d_one(xrow, vrow, wealth_point)
    )(wealth_grid.reshape((-1, nW))[flat_idx], val_sel)

    z_corner = jnp.where(constrained, v_cc, z_corner_unconstrained)

    # regular tensor-product blend
    t_b = jnp.broadcast_to(t, sel.shape)
    w_axes = jnp.where(sel == 0, 1.0 - t_b, t_b)
    w_corners = jnp.prod(w_axes, axis=1)

    return jnp.sum(w_corners * z_corner)


def _interp1d_one(xrow, vrow, xnew):
    hi, lo = get_index_high_and_low(xrow, xnew)
    lo_v, hi_v = vrow[lo], vrow[hi]
    lo_x, hi_x = xrow[lo], xrow[hi]
    denom = jnp.maximum(hi_x - lo_x, jnp.finfo(xrow.dtype).eps)
    s = (xnew - lo_x) / denom
    return lo_v + s * (hi_v - lo_v)
