"""Fedor's Upper Envelope algorithm.

Based on the original MATLAB code by Fedor Iskhakov:
https://github.com/fediskhakov/dcegm/blob/master/model_retirement.m

"""

from typing import Callable, Dict, List, Tuple

import numpy as np

EPS = 2e-16


def upper_envelope(
    endog_grid: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    state_choice_dict: Dict,
    params: Dict[str, float],
    compute_utility: Callable,
    expected_value_zero_assets: float,
    final_grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the Upper Envelope algorithm and drops sub-optimal points.

    Calculates the upper envelope over the overlapping segments of the
    decision-specific value functions, which in fact are value "correspondences"
    in this case, where multiple solutions are detected. The dominated grid
    points are then eliminated from the endogenous wealth grid.
    Discrete choices introduce kinks and non-concave regions in the value
    function that lead to discontinuities in the policy function of the
    continuous (consumption) choice. In particular, the value function has a
    non-concave region where the decision-specific values of the
    alternative discrete choices (e.g. continued work or retirement) cross.
    These are referred to as "primary" kinks.
    As a result, multiple local optima for consumption emerge and the Euler
    equation has multiple solutions.
    Moreover, these "primary" kinks propagate back in time and manifest
    themselves in an accumulation of "secondary" kinks in the choice-specific
    value functions in earlier time periods, which, in turn, also produce an
    increasing number of discontinuities in the consumption functions
    in earlier periods of the life cycle.
    These discontinuities in consumption rules in period t are caused by the
    worker's anticipation of landing exactly at the kink points in the
    subsequent periods t + 1, t + 2, ..., T under the optimal consumption policy.

    Args:
        endog_grid (np.ndarray): 1D array of shape (n_endog_wealth_grid,) containing
            the endogenous wealth grid at the beginning of period t.
        policy (np.ndarray): 1D array of shape (n_endog_wealth_grid,) containing the
            choice-specific optimal consumption at the beginning of period t.
        value (np.ndarray): 1D array of shape (n_endog_wealth_grid,) containing the
            choice-specific expected value of optimal consumption at the beginning of period t.
        state_choice_dict (Dict): Dictionary containing state and choice variables.
        params (Dict[str, float]): Dictionary containing model parameters.
        compute_utility (Callable): Function to compute utility given consumption.
        expected_value_zero_assets (float): Expected value when assets at end of period are zero.
        final_grid_size (int): Number of grid points in the final asset grid, typically
            1.2 times n_endog_wealth_grid.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - endog_grid (np.ndarray): 1D array of shape (final_grid_size,) containing the
                refined endogenous wealth grid.
            - policy (np.ndarray): 1D array of shape (final_grid_size,) containing the
                refined choice-specific policy function.
            - value (np.ndarray): 1D array of shape (final_grid_size,) containing the
                refined choice-specific value function.

    """
    # Combine grids into 2D arrays for processing
    policy = np.vstack([endog_grid, policy])
    value = np.vstack([endog_grid, value])

    # Check for credit constraint
    min_wealth = np.min(value[0, :])
    is_credit_constrained = value[0, 0] > min_wealth

    # Handle non-monotonic regions and augment grid if needed
    if is_credit_constrained:
        policy, value = _augment_grid_left(
            policy,
            value,
            state_choice_dict,
            expected_value_zero_assets,
            min_wealth,
            endog_grid.size,
            params,
            compute_utility,
        )

    non_concave_segments = locate_non_concave_regions(value)

    # Process non-concave regions if multiple segments exist
    if len(non_concave_segments) > 1:
        refined_value, kink_points = compute_upper_envelope(non_concave_segments)
        dominated_indices = find_dominated_points(value, refined_value)

        if is_credit_constrained:
            refined_value = np.hstack(
                [np.array([[0], [expected_value_zero_assets]]), refined_value]
            )

        refined_policy = refine_policy(policy, dominated_indices, kink_points)

        # Add kink points to value grid
        for i in range(kink_points.shape[1]):
            insert_idx = np.where(refined_value[0, :] < kink_points[0, i])[0].max() + 1
            refined_value = np.insert(
                refined_value, insert_idx, kink_points[:, i], axis=1
            )
    else:
        refined_value, refined_policy = value, policy

    # Prepare final arrays
    final_value = (
        np.append(expected_value_zero_assets, refined_value[1, :])
        if not is_credit_constrained
        else refined_value[1, :]
    )
    final_endog_grid = np.append(0.0, refined_policy[0, :])
    final_policy = np.append(0.0, refined_policy[1, :])

    # Initialize output arrays with NaNs
    result_endog_grid = np.full(final_grid_size, np.nan)
    result_policy = np.full(final_grid_size, np.nan)
    result_value = np.full(final_grid_size, np.nan)

    # Fill output arrays
    result_endog_grid[: len(final_endog_grid)] = final_endog_grid
    result_policy[: len(final_policy)] = final_policy
    result_value[: len(final_value)] = final_value

    return result_endog_grid, result_policy, result_value


def locate_non_concave_regions(value: np.ndarray) -> List[np.ndarray]:
    """Locates non-concave regions in the value function.

    Identifies non-monotonicity in the endogenous wealth grid where a grid point
    to the right is smaller than its preceding point, indicating the value function
    bends "backwards". Non-concave regions in the value function are reflected by
    non-monotonous regions in the underlying endogenous wealth grid. Multiple
    solutions to the Euler equation cause the standard EGM loop to produce a
    "value correspondence" rather than a value function. The elimination of
    suboptimal grid points converts this correspondence back to a proper function.

    Args:
        value (np.ndarray): 2D array of shape (2, n_endog_wealth_grid) containing the
            choice-specific value function "correspondences", where n_endog_wealth_grid
            varies based on the number of kinks and non-concave regions.

    Returns:
        List[np.ndarray]: List of arrays, each of shape (2, n_segment), containing
            segments of the value function where n_segment is the length of each
            non-monotonic segment.

    """
    segments = []
    current_value = value
    is_monotonic = current_value[0, 1:] > current_value[0, :-1]

    while True:
        non_monotonic_idx = np.where(is_monotonic != is_monotonic[0])[0]

        if not non_monotonic_idx.size:
            if segments:  # Only append if we've already found segments
                segments.append(current_value)
            break

        split_idx = min(non_monotonic_idx)
        part_one, part_two = _partition_grid(current_value, split_idx)
        segments.append(part_one)
        current_value = part_two
        is_monotonic = is_monotonic[split_idx:]

    return segments


def compute_upper_envelope(segments: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the upper envelope and refines the value function correspondence.

    Detects suboptimal points in the value function correspondence, removes them,
    and includes kink points along with their interpolated values. This process
    converts the value correspondence back to a proper function, yielding the
    refined endogenous wealth grid and value function.

    Args:
        segments (List[np.ndarray]): List of non-monotonous segments in the endogenous
            wealth grid, each of shape (2, n_segment), where n_segment is the length
            of the given non-monotonous segment.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - refined_value (np.ndarray): 2D array of shape (2, n_grid_refined) containing
                the refined endogenous wealth grid and corresponding value function.
            - kink_points (np.ndarray): 2D array of shape (2, n_intersect_points) containing
                kink points and their corresponding interpolated values.

    """
    # Create unified wealth grid
    wealth_grid = np.unique(np.concatenate([seg[0] for seg in segments]))

    # Interpolate values for each segment
    interp_values = np.array(
        [
            _linear_interpolation_with_inserting_missing_values(
                seg[0], seg[1], wealth_grid, -np.inf
            )
            for seg in segments
        ]
    )

    # Identify top segments
    max_values = np.max(interp_values, axis=0)
    top_segments = interp_values == max_values

    grid_points = [wealth_grid[0]]
    values = [interp_values[0, 0]]
    kink_points = []
    kink_values = []

    current_segment = np.where(top_segments[:, 0])[0][0]

    for i in range(1, len(wealth_grid)):
        next_segment = np.where(top_segments[:, i])[0][0]

        if next_segment != current_segment:
            grid_start, grid_end = wealth_grid[i - 1 : i + 1]
            val_seg1 = _linear_interpolation_with_inserting_missing_values(
                segments[current_segment][0],
                segments[current_segment][1],
                np.array([grid_start, grid_end]),
                np.nan,
            )
            val_seg2 = _linear_interpolation_with_inserting_missing_values(
                segments[next_segment][0],
                segments[next_segment][1],
                np.array([grid_start, grid_end]),
                np.nan,
            )

            if np.all(np.isfinite(np.vstack([val_seg1, val_seg2]))):
                seg1 = np.array([[grid_start, grid_end], val_seg1])
                seg2 = np.array([[grid_start, grid_end], val_seg2])

                try:
                    x_intersect, y_intersect = _intersection_closed_form(seg1, seg2)
                    if grid_start <= x_intersect <= grid_end:
                        all_seg_vals = np.array(
                            [
                                _linear_interpolation_with_inserting_missing_values(
                                    seg[0], seg[1], np.array([x_intersect]), -np.inf
                                )[0]
                                for seg in segments
                            ]
                        )
                        if np.max(all_seg_vals) in (
                            all_seg_vals[current_segment],
                            all_seg_vals[next_segment],
                        ):
                            grid_points.append(x_intersect)
                            values.append(y_intersect)
                            kink_points.append(x_intersect)
                            kink_values.append(y_intersect)
                except ValueError:
                    pass

        if any(abs(segments[next_segment][0] - wealth_grid[i]) < EPS):
            grid_points.append(wealth_grid[i])
            values.append(max_values[i])

        current_segment = next_segment

    refined_value = np.array([grid_points, values])
    kink_points_array = np.array([kink_points, kink_values])

    return refined_value, kink_points_array


def find_dominated_points(
    value: np.ndarray, refined_value: np.ndarray, significance: int = 10
) -> np.ndarray:
    """Identifies indices of dominated points in the value function correspondence.

    Points are considered dominated if their grid or value differs significantly
    from all points in the refined set, based on a tolerance of 2 * 10^(-significance).

    Args:
        value (np.ndarray): 2D array of shape (2, n_endog_wealth_grid) containing
            choice-specific value function correspondences.
        refined_value (np.ndarray): 2D array of shape (2, n_grid_refined) containing
            the refined value function with suboptimal points dropped and kink points added.
        significance (int): Level of significance for comparison tolerance (default: 10).

    Returns:
        np.ndarray: 1D array of shape (n_dominated_points,) containing indices of
            dominated points, where n_dominated_points varies.

    """
    tol = 2 * 10 ** (-significance)
    grid_diff = np.abs(value[0, :, None] - refined_value[0, None, :])
    value_diff = np.abs(value[1, :, None] - refined_value[1, None, :])

    is_non_dominated = np.any((grid_diff < tol) & (value_diff < tol), axis=1)
    return np.arange(value.shape[1])[~is_non_dominated]


def refine_policy(
    policy: np.ndarray, dominated_indices: np.ndarray, kink_points: np.ndarray
) -> np.ndarray:
    """Refines the policy correspondence by removing suboptimal points and adding kink
    points.

    Args:
        policy (np.ndarray): 2D array of shape (2, n_endog_wealth_grid) containing the
            choice-specific policy function correspondence.
        dominated_indices (np.ndarray): 1D array of shape (n_dominated_points,) containing
            indices of dominated points in the endogenous wealth grid.
        kink_points (np.ndarray): 2D array of shape (2, n_kink_points) containing kink
            points and their corresponding interpolated values.

    Returns:
        np.ndarray: 2D array of shape (2, n_grid_refined) containing the refined
            choice-specific policy function with suboptimal points removed and kink points added.

    """
    wealth_grid = np.delete(policy[0, :], dominated_indices)
    consumption = np.delete(policy[1, :], dominated_indices)

    for kink in kink_points[0, :]:
        left_idx = max(
            [i for i in np.where(policy[0, :] < kink)[0] if i not in dominated_indices]
        )
        right_idx = min(
            [i for i in np.where(policy[0, :] > kink)[0] if i not in dominated_indices]
        )

        interp_left = _linear_interpolation_with_extrapolation(
            policy[0, left_idx : left_idx + 2], policy[1, left_idx : left_idx + 2], kink
        )
        interp_right = _linear_interpolation_with_extrapolation(
            policy[0, right_idx - 1 : right_idx + 1],
            policy[1, right_idx - 1 : right_idx + 1],
            kink,
        )

        insert_idx = np.where(wealth_grid > kink)[0][0]
        wealth_grid = np.insert(wealth_grid, insert_idx, [kink, kink - 0.001 * EPS])
        consumption = np.insert(consumption, insert_idx, [interp_left, interp_right])

    return np.stack([wealth_grid, consumption])


def _augment_grid_left(
    policy: np.ndarray,
    value: np.ndarray,
    state_choice_dict: Dict,
    expected_value_zero_assets: float,
    min_wealth: float,
    grid_size: int,
    params: Dict,
    compute_utility: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extends the endogenous wealth grid, value, and policy function to the left.

    Args:
        policy (np.ndarray): 2D array of shape (2, n_endog_wealth_grid) containing the
            choice-specific policy function correspondence.
        value (np.ndarray): 2D array of shape (2, n_endog_wealth_grid) containing the
            choice-specific value function correspondence.
        state_choice_dict (Dict): Dictionary containing state and choice variables.
        expected_value_zero_assets (float): Expected value when wealth is zero.
        min_wealth (float): Minimum wealth level in the endogenous wealth grid.
        grid_size (int): Number of grid points in the exogenous wealth grid.
        params (Dict): Dictionary containing model parameters.
        compute_utility (Callable): Function to compute utility given consumption.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - policy_augmented (np.ndarray): 2D array of shape (2, n_grid_augmented)
                containing the augmented endogenous grid and policy function.
            - value_augmented (np.ndarray): 2D array of shape (2, n_grid_augmented)
                containing the augmented endogenous grid and value function.

    """
    extra_points = np.linspace(min_wealth, value[0, 0], grid_size // 10)
    utility = compute_utility(
        consumption=extra_points, params=params, **state_choice_dict
    )
    extra_values = utility + params["beta"] * expected_value_zero_assets

    policy_augmented = np.vstack(
        [np.append(extra_points, policy[0, 1:]), np.append(extra_points, policy[1, 1:])]
    )
    value_augmented = np.vstack(
        [np.append(extra_points, value[0, 1:]), np.append(extra_values, value[1, 1:])]
    )

    return policy_augmented, value_augmented


def _partition_grid(value: np.ndarray, split_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Splits the grid into two parts at the specified index.

    The index after which the separation occurs is also included in the second partition.

    Args:
        value (np.ndarray): 2D array of shape (2, n_endog_wealth_grid) containing the
            choice-specific value function correspondence.
        split_idx (int): Index where the endogenous wealth grid is separated.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - part_one (np.ndarray): 2D array of shape (2, split_idx+1) containing the first partition.
            - part_two (np.ndarray): 2D array of shape (2, n_endog_wealth_grid-split_idx) containing the second partition.

    """
    split_idx = min(split_idx, value.shape[1])
    return value[:, : split_idx + 1], value[:, split_idx:]


def _linear_interpolation_with_extrapolation(
    x: np.ndarray, y: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """Performs linear interpolation with extrapolation for new x-values.

    Args:
        x (np.ndarray): 1D array of shape (n,) containing the x-values.
        y (np.ndarray): 1D array of shape (n,) containing the y-values corresponding to x.
        x_new (np.ndarray): 1D array of shape (m,) or float containing the new x-values.

    Returns:
        np.ndarray: 1D array of shape (m,) or float containing the interpolated or
            extrapolated y-values corresponding to x_new.

    """
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    high_idx = np.searchsorted(x, x_new).clip(max=x.size - 1, min=1)
    low_idx = high_idx - 1

    slope = (y[high_idx] - y[low_idx]) / (x[high_idx] - x[low_idx])
    return slope * (x_new - x[low_idx]) + y[low_idx]


def _linear_interpolation_with_inserting_missing_values(
    x: np.ndarray, y: np.ndarray, x_new: np.ndarray, missing_value: float
) -> np.ndarray:
    """Performs linear interpolation, inserting missing values for out-of-range
    x-values.

    Args:
        x (np.ndarray): 1D array of shape (n,) containing the x-values.
        y (np.ndarray): 1D array of shape (n,) containing the y-values corresponding to x.
        x_new (np.ndarray): 1D array of shape (m,) or float containing the new x-values.
        missing_value (float): Value to set for x_new values outside the range of x.

    Returns:
        np.ndarray: 1D array of shape (m,) or float containing the interpolated y-values,
            with missing_value set for out-of-range x_new values.

    """
    result = _linear_interpolation_with_extrapolation(x, y, x_new)
    result[(x_new < x.min()) | (x_new > x.max())] = missing_value
    return result


def _intersection_closed_form(
    seg1: np.ndarray, seg2: np.ndarray
) -> Tuple[float, float]:
    """Finds the intersection point of two line segments in 2D space.

    Args:
        seg1 (np.ndarray): 2D array of shape (2, 2) defining the first segment's x and y coordinates.
        seg2 (np.ndarray): 2D array of shape (2, 2) defining the second segment's x and y coordinates.

    Returns:
        Tuple[float, float]: The x and y coordinates of the intersection point.

    Raises:
        ValueError: If the segments are parallel.

    """
    x1, x2 = seg1[0]
    y1, y2 = seg1[1]
    x3, x4 = seg2[0]
    y3, y4 = seg2[1]

    m1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - m1 * x1
    m2 = (y4 - y3) / (x4 - x3)
    b2 = y3 - m2 * x3

    if np.isclose(m1, m2):
        raise ValueError(
            "Two Upper Envelope Segments are parallel, no unique intersection."
        )

    x_intersect = (b2 - b1) / (m1 - m2)
    return x_intersect, m1 * x_intersect + b1
