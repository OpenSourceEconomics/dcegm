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
    # Combine grids into 3D array for processing
    initial_grid_size = endog_grid.size
    candidates = np.vstack([endog_grid, policy, value])

    # Check for credit constraint
    min_wealth = np.min(candidates[0, :])
    is_credit_constrained = candidates[0, 0] > min_wealth

    # Handle non-monotonic regions and augment grid if needed
    if is_credit_constrained:
        candidates = _augment_grid_left(
            candidates,
            state_choice_dict,
            expected_value_zero_assets,
            min_wealth,
            initial_grid_size,
            params,
            compute_utility,
        )

    non_concave_segments = locate_non_concave_regions(candidates)

    # Process non-concave regions if multiple segments exist
    if len(non_concave_segments) > 1:
        endog_grid, policy, value, kinks = compute_upper_envelope(
            non_concave_segments, final_grid_size - 1
        )

    else:
        endog_grid = np.full(final_grid_size, np.nan)[:initial_grid_size] = candidates[
            0, :
        ]
        policy = np.full(final_grid_size, np.nan)[:initial_grid_size] = candidates[1, :]
        value = np.full(final_grid_size, np.nan)[:initial_grid_size] = candidates[2, :]
        kinks = np.full((3, final_grid_size), np.nan)

    # Add point for no begin of period assets
    value = np.append(expected_value_zero_assets, value)
    endog_grid = np.append(0.0, endog_grid)
    policy = np.append(0.0, policy)

    return endog_grid, policy, value


def locate_non_concave_regions(candidates: np.ndarray) -> List[np.ndarray]:
    """Locate segments where the grid is non-monotonic."""
    diffs = candidates[0, 1:] - candidates[0, :-1]
    change_points = np.where(np.sign(diffs[:-1]) != np.sign(diffs[1:]))[0] + 1
    segments = []
    start = 0

    for idx in change_points:
        segments.append(candidates[:, start : idx + 1])
        start = idx
    segments.append(candidates[:, start:])  # Add final segment

    return segments


def compute_upper_envelope(
    segments: List[np.ndarray], grid_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the upper envelope from overlapping value function segments."""

    # 1. Create a unified wealth grid
    unified_grid = np.unique(np.concatenate([seg[0] for seg in segments]))

    # 2. Interpolate all segments over the unified grid using NumPy
    interpolated_value = np.array(
        [
            np.interp(unified_grid, seg[0], seg[2], left=-np.inf, right=-np.inf)
            for seg in segments
        ]
    )
    interpolated_policy = np.array(
        [
            np.interp(unified_grid, seg[0], seg[1], left=np.nan, right=np.nan)
            for seg in segments
        ]
    )

    # 3. Find max value and best segment at each grid point
    max_values = np.max(interpolated_value, axis=0)
    best_segment = np.argmax(interpolated_value, axis=0)

    # 4. Initialize grid, value and policy with final_grid_size + store kinks
    endog_grid = np.full(grid_size, np.nan)
    value = np.full(grid_size, np.nan)
    policy = np.full(grid_size, np.nan)
    # store every kink twice with engo_grid, value and policy
    kinks = np.full((3, grid_size), np.nan)

    # 5. Set the first point
    endog_grid[0] = unified_grid[0]
    value[0] = max_values[0]
    policy[0] = interpolated_policy[best_segment[0], 0]

    # 6. Iterate through the wealth grid to find kinks and add points
    insert_idx = 1
    for i in range(1, len(unified_grid)):
        prev_idx, curr_idx = best_segment[i - 1], best_segment[i]
        if prev_idx != curr_idx:  # best segment switches
            # endo_grid left and right of the kink
            x0, x1 = unified_grid[i - 1], unified_grid[i]
            # find the value right and left of the kink for the two value function segments
            y0_prev, y1_prev = (
                interpolated_value[prev_idx, i - 1],
                interpolated_value[prev_idx, i],
            )
            y0_curr, y1_curr = (
                interpolated_value[curr_idx, i - 1],
                interpolated_value[curr_idx, i],
            )
            # find policy right and left of the kink for the two policy function segments
            p0_prev, p1_prev = (
                interpolated_policy[prev_idx, i - 1],
                interpolated_policy[prev_idx, i],
            )
            p0_curr, p1_curr = (
                interpolated_policy[curr_idx, i - 1],
                interpolated_policy[curr_idx, i],
            )
            # only add the kink point if it is between two actual segments
            # no extrapolation for any of the segments of the value function
            if all(np.isfinite([y0_prev, y1_prev, y0_curr, y1_curr])):
                # Compute slopes and intercepts of the two segments of the value function
                slope_prev = (y1_prev - y0_prev) / (x1 - x0)
                intercept_prev = y0_prev - slope_prev * x0
                slope_curr = (y1_curr - y0_curr) / (x1 - x0)
                intercept_curr = y0_curr - slope_curr * x0
                # Calculate endo_grid at kink point
                x_kink = (intercept_curr - intercept_prev) / (slope_prev - slope_curr)
                # Calculate value at kink point
                value_kink = (
                    slope_prev * x_kink + intercept_prev
                )  # could be done with either line
                # Calculate policy left and right of the kink
                interp_left = interpolate_with_extrapolation(
                    [x0, x1], [p0_prev, p1_prev], x_kink
                )
                interp_right = interpolate_with_extrapolation(
                    [x0, x1], [p0_curr, p1_curr], x_kink
                )
                # Set kink coordinates with offset on endo_grid
                endog_grid[insert_idx : insert_idx + 2] = [x_kink - EPS, x_kink + EPS]
                # Set value at kink point on endo_grid
                value[insert_idx : insert_idx + 2] = [value_kink, value_kink]
                # Set policy at kink points on endo_grid
                policy[insert_idx : insert_idx + 2] = [interp_left, interp_right]
                # Store kink endo_grid, value and policy left and right of the kink
                kinks[:, insert_idx : insert_idx + 2] = np.array(
                    [
                        [x_kink - EPS, x_kink + EPS],
                        [value_kink, value_kink],
                        [interp_left, interp_right],
                    ]
                )
                insert_idx += 2

        # always add the dominating point
        endog_grid[insert_idx] = unified_grid[i]
        value[insert_idx] = max_values[i]
        policy[insert_idx] = interpolated_policy[best_segment[i], i]
        insert_idx += 1

    return endog_grid, policy, value, kinks


def _augment_grid_left(
    candidates: np.ndarray,
    state_choice_dict: Dict,
    expected_value_zero_assets: float,
    min_wealth: float,
    grid_size: int,
    params: Dict,
    compute_utility: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extends the endogenous wealth grid, value, and policy function to the left.

    Args:
        state_choice_dict (Dict): Dictionary containing state and choice variables.
        expected_value_zero_assets (float): Expected value when wealth is zero.
        min_wealth (float): Minimum wealth level in the endogenous wealth grid.
        grid_size (int): Number of grid points in the exogenous wealth grid.
        params (Dict): Dictionary containing model parameters.
        compute_utility (Callable): Function to compute utility given consumption.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - endog_grid (np.ndarray): Extended endogenous wealth grid.
            - policy (np.ndarray): Extended policy function.
            - extra_values (np.ndarray): Extended value function for the extra points.

    """
    extra_points = np.linspace(min_wealth, candidates[0, 0], grid_size // 10)
    utility = compute_utility(
        consumption=extra_points, params=params, **state_choice_dict
    )
    extra_values = utility + params["beta"] * expected_value_zero_assets

    return np.vstack(
        (
            np.append(extra_points, candidates[0, 1:]),
            np.append(extra_points, candidates[1, 1:]),
            np.append(extra_values, candidates[2, 1:]),
        )
    )


def interpolate_with_extrapolation(x, y, x0):
    return y[0] + ((y[1] - y[0]) / (x[1] - x[0])) * (x0 - x[0])
