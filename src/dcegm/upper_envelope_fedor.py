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

        refined_policy = refine_policy(policy, dominated_indices, kink_points)

    else:
        refined_value, refined_policy = value, policy

    # Prepare final arrays
    final_value = np.append(expected_value_zero_assets, refined_value[1, :])
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
    """Locate segments where the grid is non-monotonic."""
    diffs = value[0, 1:] - value[0, :-1]
    change_points = np.where(np.sign(diffs[:-1]) != np.sign(diffs[1:]))[0] + 1

    segments = []
    start = 0
    for idx in change_points:
        segments.append(value[:, start : idx + 1])
        start = idx
    segments.append(value[:, start:])  # Add final segment

    return segments


def compute_upper_envelope(segments: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the upper envelope from overlapping value function segments."""

    # 1. Create a unified wealth grid
    wealth_grid = np.unique(np.concatenate([seg[0] for seg in segments]))

    # 2. Interpolate all segments over the unified grid using NumPy
    interpolated = np.array(
        [
            np.interp(wealth_grid, seg[0], seg[1], left=-np.inf, right=-np.inf)
            for seg in segments
        ]
    )

    # 3. Find max value and best segment at each grid point
    max_values = np.max(interpolated, axis=0)
    best_segment = np.argmax(interpolated, axis=0)

    grid = [wealth_grid[0]]
    values = [max_values[0]]
    kinks_x, kinks_y = [], []

    for i in range(1, len(wealth_grid)):
        prev_idx, curr_idx = best_segment[i - 1], best_segment[i]

        if prev_idx != curr_idx:

            # If segments are different, find intersection point
            x0, x1 = wealth_grid[i - 1], wealth_grid[i]
            y0_prev, y1_prev = interpolated[prev_idx, i - 1], interpolated[prev_idx, i]
            y0_curr, y1_curr = interpolated[curr_idx, i - 1], interpolated[curr_idx, i]

            if all(np.isfinite([y0_prev, y1_prev, y0_curr, y1_curr])):
                # Compute slopes and intercepts
                m1 = (y1_prev - y0_prev) / (x1 - x0)
                b1 = y0_prev - m1 * x0
                m2 = (y1_curr - y0_curr) / (x1 - x0)
                b2 = y0_curr - m2 * x0
                x_kink = (b2 - b1) / (m1 - m2)
                y_kink = m1 * x_kink + b1
                grid.append(x_kink)
                grid.append(x_kink + EPS)  # Add a small offset to avoid duplicates
                values.append(y_kink)
                values.append(y_kink)
                kinks_x.append(x_kink)
                kinks_y.append(y_kink)

        if any(np.abs(segments[curr_idx][0] - wealth_grid[i]) < EPS):
            grid.append(wealth_grid[i])
            values.append(max_values[i])

    return np.array([grid, values]), np.array([kinks_x, kinks_y])


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
    """Refines the policy by removing dominated points and adding kink points."""
    wealth_grid = np.delete(policy[0, :], dominated_indices)
    consumption = np.delete(policy[1, :], dominated_indices)

    for kink in kink_points[0, :]:

        # Identify left and right indices in the original (full) policy grid
        valid_indices = [
            i for i in range(policy.shape[1]) if i not in dominated_indices
        ]
        left_candidates = [i for i in valid_indices if policy[0, i] < kink]
        right_candidates = [i for i in valid_indices if policy[0, i] > kink]

        if not left_candidates or not right_candidates:
            continue  # skip if kink is outside the valid interpolation range

        left_idx = max(left_candidates)
        right_idx = min(right_candidates)

        interp_left = interpolate_with_extrapolation(
            policy[0, left_idx : left_idx + 2], policy[1, left_idx : left_idx + 2], kink
        )
        interp_right = interpolate_with_extrapolation(
            policy[0, right_idx - 1 : right_idx + 1],
            policy[1, right_idx - 1 : right_idx + 1],
            kink,
        )

        insert_idx = np.where(wealth_grid > kink)[0][0]
        wealth_grid = np.insert(wealth_grid, insert_idx, [kink, kink + EPS])
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


def interpolate_with_extrapolation(x, y, x0):
    return y[0] + ((y[1] - y[0]) / (x[1] - x[0])) * (x0 - x[0])
