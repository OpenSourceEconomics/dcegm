"""Fedor's Upper Envelope algorithm, refactored for readability and vmap compatibility.

Based on the original MATLAB code by Fedor Iskhakov:
https://github.com/fediskhakov/dcegm/blob/master/model_retirement.m

"""

from typing import Callable, NamedTuple, Tuple

import jax.numpy as jnp
from jax import lax

EPS = 2e-16


def upper_envelope(
    endog_grid: jnp.ndarray,
    policy: jnp.ndarray,
    value: jnp.ndarray,
    state_choice_vec: jnp.ndarray,
    params: jnp.ndarray,
    compute_utility: Callable,
    expected_value_zero_assets: float,
    n_final_asset_grid: int,
    max_segments: int = 100,  # Maximum number of non-monotonic segments
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes the upper envelope over decision-specific value correspondences.

    Eliminates suboptimal points from the endogenous wealth grid and handles
    non-concave regions caused by discrete choices, which introduce kinks in the
    value function and discontinuities in the policy function.

    Args:
        endog_grid: 1D array of shape (n_grid_wealth,) with endogenous wealth grid.
        policy: 1D array of shape (n_grid_wealth,) with choice-specific policy (consumption).
        value: 1D array of shape (n_grid_wealth,) with choice-specific value correspondence.
        state_choice_vec: 1D array with state and choice variables.
        params: 1D array with model parameters (including beta for discount factor).
        compute_utility: Function to compute utility given consumption and parameters.
        expected_value_zero_assets: Expected value when end-of-period assets are zero.
        n_final_asset_grid: Fixed size of the final asset grid (typically 1.2 * n_grid_wealth).
        max_segments: Maximum number of non-monotonic segments for fixed-size arrays.

    Returns:
        Tuple of three 1D arrays of shape (n_final_asset_grid,):
        - Refined endogenous wealth grid.
        - Refined policy function (optimal consumption).
        - Refined value function.

    """
    n_grid_wealth = endog_grid.shape[0]

    # Combine grid with policy and value for processing
    policy_grid = jnp.vstack([endog_grid, policy])  # Shape: (2, n_grid_wealth)
    value_grid = jnp.vstack([endog_grid, value])  # Shape: (2, n_grid_wealth)

    # Check for credit constraint (non-monotonicity below minimum wealth)
    min_wealth = jnp.min(value_grid[0, :])
    is_credit_constrained = value_grid[0, 0] > min_wealth

    # Handle credit constraint by augmenting grid if needed
    def augment_grid_true(_):
        return _augment_grid(
            policy=policy_grid,
            value=value_grid,
            state_choice_vec=state_choice_vec,
            expected_value_zero_assets=expected_value_zero_assets,
            min_wealth=min_wealth,
            n_grid_wealth=n_grid_wealth,
            params=params,
            compute_utility=compute_utility,
        )

    def augment_grid_false(_):
        # Pad original grids to match augmented shape
        n_points = n_grid_wealth // 10
        pad_size = n_grid_wealth + n_points - 1
        padded_policy = jnp.pad(
            policy_grid,
            ((0, 0), (0, pad_size - n_grid_wealth)),
            constant_values=jnp.nan,
        )
        padded_value = jnp.pad(
            value_grid, ((0, 0), (0, pad_size - n_grid_wealth)), constant_values=jnp.nan
        )
        return padded_policy, padded_value

    policy_grid, value_grid = lax.cond(
        is_credit_constrained, augment_grid_true, augment_grid_false, operand=None
    )

    # Identify non-monotonic segments in the value function
    segments, n_segments = _locate_non_concave_regions(
        value_grid, max_segments, n_grid_wealth
    )

    # Process multiple segments if present
    if n_segments > 1:
        # Compute the upper envelope and identify kink points
        value_refined, points_to_add = _compute_upper_envelope(segments, n_segments)

        # Identify dominated points to remove
        dominated_indices = _find_dominated_points(value_grid, value_refined)

        # Adjust for credit constraint
        value_refined = lax.cond(
            is_credit_constrained,
            lambda _: jnp.hstack(
                [jnp.array([[0.0], [expected_value_zero_assets]]), value_refined]
            ),
            lambda x: x,
            operand=value_refined,
        )

        # Refine policy by removing dominated points and adding kink points
        policy_refined = _refine_policy(policy_grid, dominated_indices, points_to_add)

        # Add kink points to value grid to capture discontinuities
        value_refined = _add_kink_points_to_value(value_refined, points_to_add)
    else:
        value_refined = value_grid
        policy_refined = policy_grid

    # Finalize outputs with zero wealth point
    value_final = lax.cond(
        is_credit_constrained,
        lambda x: x[1, :],
        lambda x: jnp.append(expected_value_zero_assets, x[1, :]),
        operand=value_refined,
    )
    endog_grid_final = jnp.append(0.0, policy_refined[0, :])
    policy_final = jnp.append(0.0, policy_refined[1, :])

    # Pad outputs to fixed size with nan for invalid entries
    output_size = n_final_asset_grid
    endog_grid_out = jnp.full(output_size, jnp.nan)
    policy_out = jnp.full(output_size, jnp.nan)
    value_out = jnp.full(output_size, jnp.nan)

    valid_length = jnp.minimum(endog_grid_final.shape[0], output_size)
    endog_grid_out = endog_grid_out.at[:valid_length].set(
        endog_grid_final[:valid_length]
    )
    policy_out = policy_out.at[:valid_length].set(policy_final[:valid_length])
    value_out = value_out.at[:valid_length].set(value_final[:valid_length])

    return endog_grid_out, policy_out, value_out


class SegmentState(NamedTuple):
    segments: jnp.ndarray
    n_segments: int
    current_grid: jnp.ndarray
    current_monotonic: jnp.ndarray
    continue_loop: jnp.ndarray


def _locate_non_concave_regions(
    value_grid: jnp.ndarray, max_segments: int, n_grid_wealth: int
) -> Tuple[jnp.ndarray, int]:
    """Identifies non-monotonic segments in the value function using jax.lax.scan.

    Non-monotonic regions in the endogenous wealth grid indicate non-concave
    regions in the value function, resulting in a value correspondence.

    Args:
        value_grid: Array of shape (2, max_grid_size) with wealth grid and values.
        max_segments: Maximum number of segments to allocate.
        n_grid_wealth: Number of grid points in the original wealth grid.

    Returns:
        Tuple of:
        - segments: Array of shape (max_segments, 2, max_grid_size) with segments.
        - n_segments: Number of valid segments.

    """
    max_grid_size = value_grid.shape[1]  # Maximum grid size after augmentation
    segments = jnp.full((max_segments, 2, max_grid_size), jnp.nan)
    is_monotonic = value_grid[0, 1:] > value_grid[0, :-1]

    def scan_body(state: SegmentState, _):
        # Find the first non-monotonic index
        diff_mask = state.current_monotonic != state.current_monotonic[0]
        non_monotonic_idx = jnp.nonzero(
            diff_mask, size=n_grid_wealth, fill_value=n_grid_wealth
        )[0]
        idx = jnp.min(non_monotonic_idx)

        # Check if no non-monotonic points remain
        no_non_monotonic = jnp.all(non_monotonic_idx == n_grid_wealth)
        continue_loop = jnp.logical_not(no_non_monotonic)

        # Partition the grid if non-monotonic points exist
        def partition_true(_):
            part_one, part_two = _partition_grid(state.current_grid, idx, max_grid_size)
            return part_one, part_two

        def partition_false(_):
            # Return current grid as part_one, empty part_two
            part_one = state.current_grid
            part_two = jnp.full((2, max_grid_size), jnp.nan)
            return part_one, part_two

        part_one, part_two = lax.cond(
            continue_loop, partition_true, partition_false, operand=None
        )

        # Update segments array
        new_segments = state.segments.at[state.n_segments].set(
            lax.cond(
                continue_loop,
                lambda _: part_one,
                lambda _: state.current_grid if state.n_segments > 0 else part_one,
                operand=None,
            )
        )

        # Update state
        new_n_segments = state.n_segments + lax.cond(
            continue_loop,
            lambda _: 1,
            lambda _: jnp.where(state.n_segments > 0, 1, 0),
            operand=None,
        )
        new_current_grid = part_two
        new_current_monotonic = state.current_monotonic[idx:]

        return (
            SegmentState(
                segments=new_segments,
                n_segments=new_n_segments,
                current_grid=new_current_grid,
                current_monotonic=new_current_monotonic,
                continue_loop=continue_loop,
            ),
            None,
        )

    # Initialize scan state
    init_state = SegmentState(
        segments=segments,
        n_segments=0,
        current_grid=value_grid,
        current_monotonic=is_monotonic,
        continue_loop=jnp.array(True),
    )

    # Run scan until no non-monotonic points remain or max_segments reached
    final_state, _ = lax.scan(
        scan_body, init_state, None, length=max_segments  # Upper bound on iterations
    )

    return final_state.segments, final_state.n_segments


def _compute_upper_envelope(
    segments: jnp.ndarray, n_segments: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the upper envelope over non-monotonic segments.

    Removes suboptimal points and adds kink points to refine the value function.

    Args:
        segments: Array of shape (max_segments, 2, max_grid_size) with segments.
        n_segments: Number of valid segments.

    Returns:
        Tuple of:
        - value_refined: Array of shape (2, n_refined) with refined grid and values.
        - points_to_add: Array of shape (2, n_kinks) with kink points and values.

    """
    # Create a unified wealth grid
    valid_segments = segments[:n_segments, 0, :]
    endog_wealth_grid = jnp.unique(jnp.concatenate(valid_segments, axis=0))

    # Interpolate values for each segment
    values_interp = jnp.array(
        [
            _linear_interpolation_with_inserting_missing_values(
                x=segments[i, 0, :],
                y=segments[i, 1, :],
                x_new=endog_wealth_grid,
                missing_value=jnp.nan,
            )
            for i in range(n_segments)
        ]
    )

    # Identify top segments
    max_values = jnp.max(values_interp, axis=0)
    top_segments = values_interp == max_values[None, :]

    grid_points = [endog_wealth_grid[0]]
    values = [values_interp[0, 0]]
    kink_points = []
    kink_values = []

    first_segment_idx = jnp.where(top_segments[:, 0])[0][0]

    for i in range(1, len(endog_wealth_grid)):
        second_segment_idx = jnp.where(top_segments[:, i])[0][0]

        if second_segment_idx != first_segment_idx:
            grid1, grid2 = endog_wealth_grid[i - 1], endog_wealth_grid[i]
            values1 = _linear_interpolation_with_inserting_missing_values(
                x=segments[first_segment_idx, 0, :],
                y=segments[first_segment_idx, 1, :],
                x_new=jnp.array([grid1, grid2]),
                missing_value=jnp.nan,
            )
            values2 = _linear_interpolation_with_inserting_missing_values(
                x=segments[second_segment_idx, 0, :],
                y=segments[second_segment_idx, 1, :],
                x_new=jnp.array([grid1, grid2]),
                missing_value=jnp.nan,
            )

            if jnp.all(jnp.isfinite(jnp.vstack([values1, values2]))) and jnp.all(
                jnp.abs(values1 - values2) > 0
            ):
                seg1 = jnp.array([[grid1, grid2], values1])
                seg2 = jnp.array([[grid1, grid2], values2])
                x_intersect, y_intersect = _intersection_closed_form(seg1, seg2)

                if grid1 <= x_intersect <= grid2:
                    segment_values = jnp.array(
                        [
                            _linear_interpolation_with_inserting_missing_values(
                                x=segments[j, 0, :],
                                y=segments[j, 1, :],
                                x_new=jnp.array([x_intersect]),
                                missing_value=jnp.nan,
                            )[0]
                            for j in range(n_segments)
                        ]
                    )
                    max_segment_idx = jnp.argmax(segment_values)

                    if max_segment_idx in [first_segment_idx, second_segment_idx]:
                        grid_points.append(x_intersect)
                        values.append(y_intersect)
                        kink_points.append(x_intersect)
                        kink_values.append(y_intersect)

        if jnp.any(
            jnp.abs(segments[second_segment_idx, 0, :] - endog_wealth_grid[i]) < EPS
        ):
            grid_points.append(endog_wealth_grid[i])
            values.append(max_values[i])

        first_segment_idx = second_segment_idx

    points_to_add = jnp.vstack([kink_points, kink_values])
    value_refined = jnp.vstack([grid_points, values])
    return value_refined, points_to_add


def _find_dominated_points(
    value_grid: jnp.ndarray, value_refined: jnp.ndarray, significance: int = 10
) -> jnp.ndarray:
    """Identifies indices of dominated points in the value correspondence.

    Args:
        value_grid: Array of shape (2, max_grid_size) with original grid and values.
        value_refined: Array of shape (2, n_refined) with refined grid and values.
        significance: Tolerance level for comparison (default: 10).

    Returns:
        Array of indices of dominated points.

    """
    tol = 2 * 10 ** (-significance)
    grid_diff = jnp.abs(value_grid[0, :, None] - value_refined[0, None, :])
    value_diff = jnp.abs(value_grid[1, :, None] - value_refined[1, None, :])
    is_non_dominated = jnp.any((grid_diff < tol) & (value_diff < tol), axis=1)
    return jnp.arange(value_grid.shape[1])[~is_non_dominated]


def _refine_policy(
    policy_grid: jnp.ndarray, dominated_indices: jnp.ndarray, points_to_add: jnp.ndarray
) -> jnp.ndarray:
    """Refines policy by removing dominated points and adding kink points.

    Args:
        policy_grid: Array of shape (2, max_grid_size) with grid and policy.
        dominated_indices: Indices of points to remove.
        points_to_add: Array of shape (2, n_kinks) with kink points and values.

    Returns:
        Refined policy array of shape (2, n_refined).

    """
    endog_grid = jnp.delete(policy_grid[0, :], dominated_indices)
    consumption = jnp.delete(policy_grid[1, :], dominated_indices)

    for x, y_right in points_to_add.T:
        left_idx = jnp.max(
            jnp.where(
                (endog_grid < x)
                & (~jnp.isin(jnp.arange(endog_grid.size), dominated_indices))
            )[0]
        )
        right_idx = jnp.min(
            jnp.where(
                (endog_grid > x)
                & (~jnp.isin(jnp.arange(endog_grid.size), dominated_indices))
            )[0]
        )

        left_y = _linear_interpolation_with_extrapolation(
            x=endog_grid[left_idx : left_idx + 2],
            y=consumption[left_idx : left_idx + 2],
            x_new=x,
        )
        right_y = _linear_interpolation_with_extrapolation(
            x=endog_grid[right_idx - 1 : right_idx + 1],
            y=consumption[right_idx - 1 : right_idx + 1],
            x_new=x,
        )

        insert_idx = jnp.where(endog_grid > x)[0][0]
        endog_grid = jnp.insert(endog_grid, insert_idx, [x, x - 0.001 * EPS])
        consumption = jnp.insert(consumption, insert_idx, [left_y, right_y])

    return jnp.vstack([endog_grid, consumption])


def _add_kink_points_to_value(
    value_grid: jnp.ndarray, points_to_add: jnp.ndarray
) -> jnp.ndarray:
    """Adds kink points to the value grid to capture discontinuities.

    Args:
        value_grid: Array of shape (2, max_grid_size) with wealth grid and values.
        points_to_add: Array of shape (2, n_kinks) with kink points and values.

    Returns:
        Updated value grid with kink points.

    """
    for x, y in points_to_add.T:
        insert_idx = jnp.max(jnp.where(value_grid[0, :] < x)[0]) + 1
        value_grid = jnp.insert(value_grid, insert_idx, jnp.array([x, y]), axis=1)
    return value_grid


def _augment_grid(
    policy: jnp.ndarray,
    value: jnp.ndarray,
    state_choice_vec: jnp.ndarray,
    expected_value_zero_assets: float,
    min_wealth: float,
    n_grid_wealth: int,
    params: jnp.ndarray,
    compute_utility: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extends wealth grid, policy, and value function to the left for credit
    constraints.

    Args:
        policy: Array of shape (2, n_grid_wealth) with grid and policy.
        value: Array of shape (2, n_grid_wealth) with grid and values.
        state_choice_vec: Array with state and choice variables.
        expected_value_zero_assets: Expected value at zero assets.
        min_wealth: Minimum wealth level in the grid.
        n_grid_wealth: Number of grid points.
        params: Array of model parameters.
        compute_utility: Function to compute utility.

    Returns:
        Tuple of augmented policy and value arrays, padded to fixed size with nan.

    """
    n_points = n_grid_wealth // 10
    grid_points = jnp.linspace(min_wealth, value[0, 0], n_points)
    utility = compute_utility(
        consumption=grid_points, params=params, **state_choice_vec
    )
    values = utility + params["beta"] * expected_value_zero_assets

    # Compute final size and pad with nan
    final_size = n_grid_wealth + n_points - 1
    policy_augmented = jnp.vstack(
        [jnp.append(grid_points, policy[0, 1:]), jnp.append(grid_points, policy[1, 1:])]
    )
    value_augmented = jnp.vstack(
        [jnp.append(grid_points, value[0, 1:]), jnp.append(values, value[1, 1:])]
    )

    # Pad to ensure consistent shape
    policy_augmented = jnp.pad(
        policy_augmented,
        ((0, 0), (0, final_size - policy_augmented.shape[1])),
        constant_values=jnp.nan,
    )
    value_augmented = jnp.pad(
        value_augmented,
        ((0, 0), (0, final_size - value_augmented.shape[1])),
        constant_values=jnp.nan,
    )

    return policy_augmented, value_augmented


def _partition_grid(
    value_grid: jnp.ndarray, split_idx: jnp.ndarray, max_grid_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Splits the value grid into two parts at the given index using dynamic slicing.

    Args:
        value_grid: Array of shape (2, max_grid_size) with grid and values.
        split_idx: Scalar index to split the grid (dynamic).
        max_grid_size: Maximum size of the grid dimension.

    Returns:
        Tuple of two arrays of shape (2, max_grid_size):
        - Left partition (padded with nan).
        - Right partition (padded with nan).

    """
    # Ensure split_idx is valid
    split_idx = jnp.minimum(split_idx, value_grid.shape[1] - 1)

    # Extract left partition: [:, :split_idx + 1]
    left_size = split_idx + 1
    part_one = lax.dynamic_slice(
        value_grid, start_indices=(0, 0), slice_sizes=(2, left_size)
    )
    part_one = jnp.pad(
        part_one, ((0, 0), (0, max_grid_size - left_size)), constant_values=jnp.nan
    )

    # Extract right partition: [:, split_idx:]
    right_size = value_grid.shape[1] - split_idx
    part_two = lax.dynamic_slice(
        value_grid, start_indices=(0, split_idx), slice_sizes=(2, right_size)
    )
    part_two = jnp.pad(
        part_two, ((0, 0), (0, max_grid_size - right_size)), constant_values=jnp.nan
    )

    return part_one, part_two


def _linear_interpolation_with_extrapolation(x: jnp.ndarray, y: jnp.ndarray, x_new):
    """Linear interpolation with extrapolation for new x values.

    Args:
        x: 1D array of x-values.
        y: 1D array of y-values.
        x_new: Scalar or array of new x-values.

    Returns:
        Interpolated y-values with extrapolation.

    """
    ind = jnp.argsort(x)
    x = x[ind]
    y = y[ind]

    ind_high = jnp.searchsorted(x, x_new).clip(max=x.shape[0] - 1, min=1)
    ind_low = ind_high - 1

    y_high = y[ind_high]
    y_low = y[ind_low]
    x_high = x[ind_high]
    x_low = x[ind_low]

    slope = (y_high - y_low) / (x_high - x_low)
    return y_low + slope * (x_new - x_low)


def _linear_interpolation_with_inserting_missing_values(
    x: jnp.ndarray, y: jnp.ndarray, x_new, missing_value
):
    """Linear interpolation with missing values for out-of-range x_new.

    Args:
        x: 1D array of x-values.
        y: 1D array of y-values.
        x_new: Scalar or array of new x-values.
        missing_value: Value to use for out-of-range x_new.

    Returns:
        Interpolated y-values with missing values for out-of-range points.

    """
    result = _linear_interpolation_with_extrapolation(x, y, x_new)
    mask = (x_new < x.min()) | (x_new > x.max())
    return jnp.where(mask, missing_value, result)


def _intersection_closed_form(
    seg1: jnp.ndarray, seg2: jnp.ndarray
) -> Tuple[float, float]:
    """Finds the intersection point of two 2D line segments.

    Args:
        seg1: Array of shape (2, 2) defining first segment (x, y coordinates).
        seg2: Array of shape (2, 2) defining second segment.

    Returns:
        Tuple of (x, y) intersection coordinates.

    Raises:
        ValueError: If segments are parallel.

    """
    x1, x2 = seg1[0]
    y1, y2 = seg1[1]
    x3, x4 = seg2[0]
    y3, y4 = seg2[1]

    m1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - m1 * x1
    m2 = (y4 - y3) / (x4 - x3)
    b2 = y3 - m2 * x3

    if jnp.isclose(m1, m2):
        raise ValueError("Segments are parallel, no unique intersection.")

    x_intersect = (b2 - b1) / (m1 - m2)
    y_intersect = m1 * x_intersect + b1
    return x_intersect, y_intersect
