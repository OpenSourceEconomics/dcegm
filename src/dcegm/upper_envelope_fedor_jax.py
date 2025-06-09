"""JAX-compatible version of Fedor's Upper Envelope algorithm."""

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

EPS = 2e-10


def upper_envelope(
    endog_grid: jnp.ndarray,
    policy: jnp.ndarray,
    value: jnp.ndarray,
    state_choice_dict: Dict,
    params: Dict[str, float],
    compute_utility: Callable,
    expected_value_zero_assets: float,
    final_grid_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX-compatible upper envelope algorithm.

    Args:
        endog_grid: 1D array of endogenous wealth grid
        policy: 1D array of choice-specific optimal consumption
        value: 1D array of choice-specific expected value
        state_choice_dict: Dictionary containing state and choice variables
        params: Dictionary containing model parameters
        compute_utility: Function to compute utility given consumption
        expected_value_zero_assets: Expected value when assets are zero
        final_grid_size: Number of grid points in final asset grid

    Returns:
        Tuple of (endog_grid, policy, value) arrays of size final_grid_size + 1

    """

    initial_grid_size = endog_grid.shape[0]
    max_n_segments = (
        initial_grid_size // 12
    )  # Maximum number of segments TODO find heuristic based on period, n_choice, taste shock scale & ask Max if based on period is allowed
    max_s_segments = (
        initial_grid_size  # Maximum size of each segment TODO think about this
    )
    n_extra = initial_grid_size // 10  # Points to add if credit constrained

    # Stack grids for processing
    candidates = jnp.vstack([endog_grid, policy, value])

    # Check for credit constraint
    min_wealth = jnp.min(candidates[0, :])
    is_credit_constrained = candidates[0, 0] > min_wealth

    # last valid index
    last_valid_index = jax.lax.cond(
        is_credit_constrained,
        lambda x: x + n_extra - 1,
        lambda x: x - 1,
        initial_grid_size,
    )

    # Augment grid if credit constrained - pad to final_grid_size for consistent shape
    candidates_padded = jnp.full((3, final_grid_size), jnp.nan)

    candidates_padded = jax.lax.cond(
        is_credit_constrained,
        lambda x: _augment_grid_left_jax(
            x,
            candidates,
            state_choice_dict,
            expected_value_zero_assets,
            min_wealth,
            n_extra,
            initial_grid_size,
            params,
            compute_utility,
        ),
        lambda x: x.at[:, :initial_grid_size].set(candidates),
        candidates_padded,
    )

    # Locate segments of fixed-size padded array
    segments_info = _locate_segments_jax(
        candidates_padded, max_n_segments, last_valid_index
    )
    n_segments = segments_info["n_segments"]

    # Process segments
    endog_out, policy_out, value_out = jax.lax.cond(
        n_segments > 1,
        lambda: _compute_upper_envelope_jax(
            candidates_padded,
            segments_info,
            final_grid_size - 1,
            max_n_segments,
            max_s_segments,
        ),
        lambda: _handle_single_segment_jax(candidates, final_grid_size - 1),
    )

    # Add point for zero begin-of-period assets and zero end-of-period assets
    endog_final = jnp.concatenate([jnp.array([0.0]), endog_out])
    policy_final = jnp.concatenate([jnp.array([0.0]), policy_out])
    value_final = jnp.concatenate([jnp.array([expected_value_zero_assets]), value_out])

    return endog_final, policy_final, value_final


def _locate_segments_jax(
    candidates_padded: jnp.ndarray, max_segments: int, last_valid_index: int
) -> Dict:
    """Locate non-concave segments using dynamic slicing.

    Args:
        candidates_padded: 2D array of shape (3, n) with wealth, policy, and value
        max_segments: Maximum number of segments to find
    Returns:
        Dictionary with segment information:
            - n_segments: Number of segments found
            - segment_starts: 1D array of segment start indices
            - segment_ends: 1D array of segment end indices

    """
    wealth = candidates_padded[0, :]
    diffs = wealth[1:] - wealth[:-1]

    # Find sign changes (potential segment boundaries)
    signs = jnp.sign(diffs)
    sign_changes = jnp.abs(signs[1:] - signs[:-1]) > EPS

    # Get change points (add 1 to account for diff indexing)
    change_indices = jnp.nonzero(sign_changes, size=max_segments, fill_value=-2)[0] + 1

    # Count how many are valid (not -1) add 1 for first segment
    n_segments = jnp.sum(change_indices >= 0) + 1

    # Compose segment start and end arrays
    segment_starts = jnp.concatenate([jnp.array([0]), change_indices[:-1]])
    segment_ends = change_indices.at[n_segments - 1].set(last_valid_index)

    return {
        "n_segments": n_segments,
        "segment_starts": segment_starts,
        "segment_ends": segment_ends,
    }


def _compute_upper_envelope_jax(
    candidates: jnp.ndarray,
    segments_info: Dict,
    final_grid_size: int,
    max_n_segments: int,
    max_s_segments: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute upper envelope from segments using JAX operations."""

    # Create unified wealth grid from all segments
    wealth_points = candidates[0, :]
    unified_grid = jax.lax.sort(
        jnp.unique(wealth_points, size=final_grid_size, fill_value=jnp.nan)
    )

    n_segments = segments_info["n_segments"]
    starts = segments_info["segment_starts"]
    ends = segments_info["segment_ends"]

    # Pre-allocate interpolation arrays
    interpolated_value = jnp.full((max_n_segments, unified_grid.shape[0]), -jnp.inf)
    interpolated_policy = jnp.full((max_n_segments, unified_grid.shape[0]), -jnp.inf)
    # Before the fori_loop, get unified grid length (non-NaN entries)
    unified_grid_len = jnp.sum(jnp.isfinite(unified_grid))

    def interpolate_segment(i, arrays):
        interp_val, interp_pol = arrays

        start_idx = starts[i]
        end_idx = ends[i] + 1
        seg_length = end_idx - start_idx

        valid_segment = (i < n_segments) & (start_idx >= 0)

        # Pad segment and mask
        seg_wealth_padded = jax.lax.dynamic_slice(
            jnp.concatenate([candidates[0, :], jnp.full(max_s_segments, jnp.nan)]),
            (start_idx,),
            (max_s_segments,),
        )
        seg_policy_padded = jax.lax.dynamic_slice(
            jnp.concatenate([candidates[1, :], jnp.full(max_s_segments, jnp.nan)]),
            (start_idx,),
            (max_s_segments,),
        )
        seg_value_padded = jax.lax.dynamic_slice(
            jnp.concatenate([candidates[2, :], jnp.full(max_s_segments, jnp.nan)]),
            (start_idx,),
            (max_s_segments,),
        )

        def interpolate():
            new_val = fast_interp_sorted_nan_safe(
                seg_wealth_padded,
                seg_value_padded,
                unified_grid,
                seg_len=seg_length,
                seg_len_new=unified_grid_len,
            )
            new_pol = fast_interp_sorted_nan_safe(
                seg_wealth_padded,
                seg_policy_padded,
                unified_grid,
                seg_len=seg_length,
                seg_len_new=unified_grid_len,
            )
            return new_val, new_pol

        new_val, new_pol = jax.lax.cond(
            valid_segment,
            interpolate,
            lambda: (
                jnp.full_like(unified_grid, -jnp.inf),
                jnp.full_like(unified_grid, -jnp.inf),
            ),
        )

        interp_val = interp_val.at[i].set(new_val)
        interp_pol = interp_pol.at[i].set(new_pol)

        return interp_val, interp_pol

    interpolated_value, interpolated_policy = jax.lax.fori_loop(
        0,
        max_n_segments,
        interpolate_segment,
        (interpolated_value, interpolated_policy),
    )

    # Find best segment at each point
    max_values = jnp.nanmax(interpolated_value, axis=0)
    best_segment = jnp.nanargmax(interpolated_value, axis=0)

    # Initialize output arrays
    endog_grid = jnp.full(final_grid_size, jnp.nan)
    value_out = jnp.full(final_grid_size, jnp.nan)
    policy_out = jnp.full(final_grid_size, jnp.nan)

    # Set first point
    endog_grid = endog_grid.at[0].set(unified_grid[0])
    value_out = value_out.at[0].set(max_values[0])
    policy_out = policy_out.at[0].set(interpolated_policy[best_segment[0], 0])

    # Process remaining points to find kinks
    def process_point(i, state):
        grid, val, pol, insert_idx = state

        prev_best = best_segment[i - 1]
        curr_best = best_segment[i]

        # Check if segment switches (kink point)
        is_kink = prev_best != curr_best

        def add_kink():
            # Calculate kink point
            x0, x1 = unified_grid[i - 1], unified_grid[i]

            # Values for both segments
            y0_prev = interpolated_value[prev_best, i - 1]
            y1_prev = interpolated_value[prev_best, i]
            y0_curr = interpolated_value[curr_best, i - 1]
            y1_curr = interpolated_value[curr_best, i]

            # Policies for both segments
            p0_prev = interpolated_policy[prev_best, i - 1]
            p1_prev = interpolated_policy[prev_best, i]
            p0_curr = interpolated_policy[curr_best, i - 1]
            p1_curr = interpolated_policy[curr_best, i]

            # Check if all values are finite
            all_finite = jnp.all(
                jnp.isfinite(jnp.array([y0_prev, y1_prev, y0_curr, y1_curr]))
            )

            def compute_kink():
                # Calculate intersection point
                slope_prev = (y1_prev - y0_prev) / (x1 - x0)
                intercept_prev = y0_prev - slope_prev * x0
                slope_curr = (y1_curr - y0_curr) / (x1 - x0)
                intercept_curr = y0_curr - slope_curr * x0

                x_kink = (intercept_curr - intercept_prev) / (slope_prev - slope_curr)
                value_kink = slope_prev * x_kink + intercept_prev

                # Interpolate policies at kink
                policy_left = p0_prev + ((p1_prev - p0_prev) / (x1 - x0)) * (
                    x_kink - x0
                )
                policy_right = p0_curr + ((p1_curr - p0_curr) / (x1 - x0)) * (
                    x_kink - x0
                )

                # Use dynamic_update_slice to update arrays at insert_idx
                new_grid = jax.lax.dynamic_update_slice(
                    grid, jnp.array([x_kink - EPS, x_kink + EPS]), [insert_idx]
                )
                new_val = jax.lax.dynamic_update_slice(
                    val, jnp.array([value_kink, value_kink]), [insert_idx]
                )
                new_pol = jax.lax.dynamic_update_slice(
                    pol, jnp.array([policy_left, policy_right]), [insert_idx]
                )

                return new_grid, new_val, new_pol, insert_idx + 2

            return jax.lax.cond(
                all_finite, compute_kink, lambda: (grid, val, pol, insert_idx)
            )

        # Add kink if needed
        grid, val, pol, new_insert_idx = jax.lax.cond(
            is_kink, add_kink, lambda: (grid, val, pol, insert_idx)
        )

        # Add current dominating point
        grid = grid.at[new_insert_idx].set(unified_grid[i])
        val = val.at[new_insert_idx].set(max_values[i])
        pol = pol.at[new_insert_idx].set(interpolated_policy[curr_best, i])

        return grid, val, pol, new_insert_idx + 1

    endog_grid, value_out, policy_out, _ = jax.lax.fori_loop(
        1, unified_grid.shape[0], process_point, (endog_grid, value_out, policy_out, 1)
    )

    return endog_grid, policy_out, value_out


def _handle_single_segment_jax(
    candidates: jnp.ndarray, final_grid_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Handle case with single segment."""
    initial_size = candidates.shape[1]

    endog_grid = (
        jnp.full(final_grid_size, jnp.nan).at[:initial_size].set(candidates[0, :])
    )
    policy = jnp.full(final_grid_size, jnp.nan).at[:initial_size].set(candidates[1, :])
    value = jnp.full(final_grid_size, jnp.nan).at[:initial_size].set(candidates[2, :])

    return endog_grid, policy, value


def _augment_grid_left_jax(
    candidates_padded: jnp.ndarray,
    candidates: jnp.ndarray,
    state_choice_dict: Dict,
    expected_value_zero_assets: float,
    min_wealth: float,
    n_extra: int,
    initial_grid_size: int,
    params: Dict,
    compute_utility: Callable,
) -> jnp.ndarray:
    """Augment grid to the left for credit constrained case."""

    extra_points = jnp.linspace(min_wealth, candidates[0, 0], n_extra)

    # Compute utility for extra points
    utility = compute_utility(
        consumption=extra_points, params=params, **state_choice_dict
    )
    extra_values = utility + params["beta"] * expected_value_zero_assets

    # Fill with augmented data
    candidates_padded = candidates_padded.at[0, :n_extra].set(extra_points)
    candidates_padded = candidates_padded.at[1, :n_extra].set(extra_points)
    candidates_padded = candidates_padded.at[2, :n_extra].set(extra_values)
    # Shift existing candidates to the right
    candidates_padded = candidates_padded.at[
        :, n_extra : initial_grid_size + n_extra
    ].set(candidates)
    return candidates_padded


def fast_interp_sorted_nan_safe(
    x: jnp.ndarray, y: jnp.ndarray, x_new: jnp.ndarray, seg_len: int, seg_len_new: int
) -> jnp.ndarray:
    """Interpolates x/y at x_new assuming:

    - x and x_new are sorted.
    - Only the first `seg_len_x` of x/y and `seg_len_new` of x_new are valid.
    - Returns -inf after first NaN or if x_new is outside valid x.

    """
    y_out_init = jnp.full_like(x_new, -jnp.inf)

    def loop_fn(i, state):
        idx, y_out, invalid = state

        x_new_i = x_new[i]
        new_invalid = invalid | (i >= seg_len_new)

        def valid_branch(state):
            idx, y_out, _ = state

            # Move right while x[idx+1] <= x_new[i] and idx < seg_len_x - 2
            idx = jax.lax.while_loop(
                lambda j: (j < seg_len - 2) & (x[j + 1] <= x_new_i),
                lambda j: j + 1,
                idx,
            )

            x0, x1 = x[idx], x[idx + 1]
            y0, y1 = y[idx], y[idx + 1]

            # Linear interpolation
            y_i = y0 + (y1 - y0) * (x_new_i - x0) / (x1 - x0)

            # Out-of-domain → -inf
            y_i = jnp.where(
                (x_new_i < x[0]) | (x_new_i > x[seg_len - 1]), -jnp.inf, y_i
            )

            y_out = y_out.at[i].set(y_i)
            return (idx, y_out, new_invalid)

        def invalid_branch(state):
            idx, y_out, _ = state
            # leave y_out[i] = -inf
            return (idx, y_out, new_invalid)

        return jax.lax.cond(new_invalid, invalid_branch, valid_branch, state)

    # Initial state: idx = 0, output -inf, valid
    _, y_out_final, _ = jax.lax.fori_loop(
        0, x_new.shape[0], loop_fn, (0, y_out_init, False)
    )

    return y_out_final
