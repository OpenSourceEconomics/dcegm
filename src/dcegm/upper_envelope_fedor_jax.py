"""JAX-compatible version of Fedor's Upper Envelope algorithm."""

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

EPS = 2e-16


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

    # Stack grids for processing
    candidates = jnp.vstack([endog_grid, policy, value])

    # Check for credit constraint
    min_wealth = jnp.min(candidates[0, :])
    is_credit_constrained = candidates[0, 0] > min_wealth

    # Augment grid if credit constrained - pad to ensure consistent shape
    max_augmented_size = int(initial_grid_size * 1.1)  # Based on your bound
    candidates_padded = jnp.full((3, max_augmented_size), jnp.nan)
    candidates_padded = candidates_padded.at[:, :initial_grid_size].set(candidates)

    candidates = jax.lax.cond(
        is_credit_constrained,
        lambda x: _augment_grid_left_jax(
            x,
            state_choice_dict,
            expected_value_zero_assets,
            min_wealth,
            initial_grid_size,
            params,
            compute_utility,
            max_augmented_size,
        ),
        lambda x: x,
        candidates_padded,
    )

    # Locate segments using fixed-size arrays
    segments_info = _locate_segments_jax(candidates, initial_grid_size)
    n_segments = segments_info["n_segments"]

    # Process segments
    endog_out, policy_out, value_out = jax.lax.cond(
        n_segments > 1,
        lambda: _compute_upper_envelope_jax(
            candidates, segments_info, final_grid_size - 1
        ),
        lambda: _handle_single_segment_jax(candidates, final_grid_size - 1),
    )

    # Add point for zero begin-of-period assets
    endog_final = jnp.concatenate([jnp.array([0.0]), endog_out])
    policy_final = jnp.concatenate([jnp.array([0.0]), policy_out])
    value_final = jnp.concatenate([jnp.array([expected_value_zero_assets]), value_out])

    return endog_final, policy_final, value_final


def _locate_segments_jax(candidates: jnp.ndarray, max_size: int) -> Dict:
    """Locate non-concave segments using fixed-size arrays."""
    wealth = candidates[0, :]
    diffs = wealth[1:] - wealth[:-1]

    # Find sign changes (potential segment boundaries)
    signs = jnp.sign(diffs)
    sign_changes = jnp.abs(signs[1:] - signs[:-1]) > EPS

    # Get change points (add 1 to account for diff indexing)
    change_indices = jnp.where(sign_changes, size=max_size // 10, fill_value=-1)[0] + 1

    # Count actual segments
    n_segments = jnp.sum(change_indices >= 0) + 1

    # Create segment start/end indices without boolean indexing
    max_segments = max_size // 10 + 1

    # Initialize segment arrays
    segment_starts = jnp.full(max_segments, -1)
    segment_ends = jnp.full(max_segments, -1)

    # Set first segment start
    segment_starts = segment_starts.at[0].set(0)

    # Count valid changes
    n_valid = jnp.sum(change_indices >= 0)

    # Use a loop to fill valid indices to avoid dynamic slicing
    def fill_segments(i, state):
        starts, ends = state
        # Only process if we have a valid change index
        is_valid = (i < n_valid) & (change_indices[i] >= 0)

        # Update segment starts (i+1 because first start is 0)
        starts = jax.lax.cond(
            is_valid, lambda: starts.at[i + 1].set(change_indices[i]), lambda: starts
        )

        # Update segment ends
        ends = jax.lax.cond(
            is_valid, lambda: ends.at[i].set(change_indices[i]), lambda: ends
        )

        return starts, ends

    # Fill segment arrays using loop
    segment_starts, segment_ends = jax.lax.fori_loop(
        0, max_segments - 1, fill_segments, (segment_starts, segment_ends)
    )

    # Set final segment end using loop to avoid dynamic indexing
    def set_final_end(i, ends):
        # Set end at position n_valid if i == n_valid
        should_set = i == n_valid
        return jax.lax.cond(
            should_set, lambda: ends.at[i].set(wealth.shape[0]), lambda: ends
        )

    segment_ends = jax.lax.fori_loop(0, max_segments, set_final_end, segment_ends)

    return {
        "n_segments": n_segments,
        "segment_starts": segment_starts,
        "segment_ends": segment_ends,
    }


def _compute_upper_envelope_jax(
    candidates: jnp.ndarray, segments_info: Dict, grid_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute upper envelope from segments using JAX operations."""

    # Create unified wealth grid from all segments
    wealth_points = candidates[0, :]
    unified_grid = jnp.unique(wealth_points, size=int(wealth_points.shape[0] * 1.1))

    n_segments = segments_info["n_segments"]
    starts = segments_info["segment_starts"]
    ends = segments_info["segment_ends"]

    # Pre-allocate interpolation arrays
    max_segments = starts.shape[0]
    interpolated_value = jnp.full((max_segments, unified_grid.shape[0]), -jnp.inf)
    interpolated_policy = jnp.full((max_segments, unified_grid.shape[0]), jnp.nan)

    # Interpolate each segment
    def interpolate_segment(i, arrays):
        interp_val, interp_pol = arrays

        # Get segment boundaries
        start_idx = starts[i]
        end_idx = ends[i]

        # Only process valid segments
        valid_segment = (i < n_segments) & (start_idx >= 0) & (end_idx > start_idx)

        # Extract segment data using dynamic slice with fixed size
        max_seg_length = int(grid_size * 1.1)  # Based on your bound - fixed size

        # Use dynamic slice with fixed segment size
        seg_wealth_padded = jax.lax.dynamic_slice(
            jnp.concatenate([candidates[0, :], jnp.full(max_seg_length, jnp.nan)]),
            (start_idx,),
            (max_seg_length,),
        )
        seg_policy_padded = jax.lax.dynamic_slice(
            jnp.concatenate([candidates[1, :], jnp.full(max_seg_length, jnp.nan)]),
            (start_idx,),
            (max_seg_length,),
        )
        seg_value_padded = jax.lax.dynamic_slice(
            jnp.concatenate([candidates[2, :], jnp.full(max_seg_length, jnp.nan)]),
            (start_idx,),
            (max_seg_length,),
        )

        # Mask out invalid entries beyond actual segment length
        seg_length = end_idx - start_idx
        valid_mask = jnp.arange(max_seg_length) < seg_length

        seg_wealth = jnp.where(valid_mask, seg_wealth_padded, jnp.nan)
        seg_policy = jnp.where(valid_mask, seg_policy_padded, jnp.nan)
        seg_value = jnp.where(valid_mask, seg_value_padded, jnp.nan)

        # Interpolate over unified grid
        new_val = jax.lax.cond(
            valid_segment,
            lambda: jnp.interp(
                unified_grid, seg_wealth, seg_value, left=-jnp.inf, right=-jnp.inf
            ),
            lambda: jnp.full_like(unified_grid, -jnp.inf),
        )

        new_pol = jax.lax.cond(
            valid_segment,
            lambda: jnp.interp(
                unified_grid, seg_wealth, seg_policy, left=jnp.nan, right=jnp.nan
            ),
            lambda: jnp.full_like(unified_grid, jnp.nan),
        )

        interp_val = interp_val.at[i].set(new_val)
        interp_pol = interp_pol.at[i].set(new_pol)

        return interp_val, interp_pol

    interpolated_value, interpolated_policy = jax.lax.fori_loop(
        0, max_segments, interpolate_segment, (interpolated_value, interpolated_policy)
    )

    # Find best segment at each point
    max_values = jnp.max(interpolated_value, axis=0)
    best_segment = jnp.argmax(interpolated_value, axis=0)

    # Initialize output arrays
    endog_grid = jnp.full(grid_size, jnp.nan)
    value_out = jnp.full(grid_size, jnp.nan)
    policy_out = jnp.full(grid_size, jnp.nan)

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
    candidates: jnp.ndarray,
    state_choice_dict: Dict,
    expected_value_zero_assets: float,
    min_wealth: float,
    initial_grid_size: int,
    params: Dict,
    compute_utility: Callable,
    max_size: int,
) -> jnp.ndarray:
    """Augment grid to the left for credit constrained case."""

    n_extra = initial_grid_size // 10
    extra_points = jnp.linspace(min_wealth, candidates[0, 0], n_extra)

    # Compute utility for extra points
    utility = compute_utility(
        consumption=extra_points, params=params, **state_choice_dict
    )
    extra_values = utility + params["beta"] * expected_value_zero_assets

    # Create output array with same shape as input
    output = jnp.full((3, max_size), jnp.nan)

    # Fill with augmented data
    total_size = (
        n_extra + initial_grid_size - 1
    )  # -1 because we skip first original point
    new_wealth = jnp.concatenate([extra_points, candidates[0, 1:initial_grid_size]])
    new_policy = jnp.concatenate([extra_points, candidates[1, 1:initial_grid_size]])
    new_value = jnp.concatenate([extra_values, candidates[2, 1:initial_grid_size]])

    output = output.at[0, :total_size].set(new_wealth)
    output = output.at[1, :total_size].set(new_policy)
    output = output.at[2, :total_size].set(new_value)

    return output
