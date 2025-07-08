"""Fedor's Upper Envelope algorithm.

Based on the original MATLAB code by Fedor Iskhakov:
https://github.com/fediskhakov/dcegm/blob/master/model_retirement.m

"""

import random
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
    n_choices: int,
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

    # if state_choice_dict == {'choice': np.uint8(0), 'dummy_stochastic': np.uint8(0), 'lagged_choice': np.uint8(0), 'period': np.uint8(14)}:
    #     breakpoint()

    # min_wealth
    min_wealth = np.min(candidates[0, :])
    # average grid step
    random.seed(101)
    rnd_indices = random.sample(
        range(candidates.shape[1]), min(15, candidates.shape[1])
    )
    avg_grid_step = np.nanmean(np.diff(candidates[0, rnd_indices]))

    # augment grid left in case of credit constraint this
    # is necessary, otherwise no drawback expect computational cost
    # do it always, for easier parallelization in the future
    # also add zero beginning of the period wealth value and policy
    candidates = _augment_grid_left(
        candidates,
        state_choice_dict,
        expected_value_zero_assets,
        min_wealth,
        avg_grid_step,
        initial_grid_size,
        params,
        compute_utility,
    )

    non_concave_segments = locate_non_concave_regions(candidates)

    # Process non-concave regions if multiple segments exist
    if len(non_concave_segments) > 1:
        endog_grid, policy, value, kinks = compute_upper_envelope(
            non_concave_segments, final_grid_size
        )

    else:
        candidates_size = candidates.shape[1]
        endog_grid = np.pad(
            candidates[0, :],
            (0, final_grid_size - candidates_size),
            mode="constant",
            constant_values=np.nan,
        )
        policy = np.pad(
            candidates[1, :],
            (0, final_grid_size - candidates_size),
            mode="constant",
            constant_values=np.nan,
        )
        value = np.pad(
            candidates[2, :],
            (0, final_grid_size - candidates_size),
            mode="constant",
            constant_values=np.nan,
        )
        kinks = np.full((3, 2), np.nan)

    return endog_grid, policy, value


def locate_non_concave_regions(candidates: np.ndarray) -> List[np.ndarray]:
    """Locate segments where the grid is non-monotonic."""

    diffs = candidates[0, 1:] - candidates[0, :-1]
    change_points = (np.where(np.sign(diffs[:-1]) != np.sign(diffs[1:]))[0] + 1)[0::2]
    regions = []
    if len(change_points) == 0:
        return regions
    else:
        start = 0
        for idx in change_points:
            regions.append(candidates[:, start : idx + 1])
            start = idx + 1
        regions.append(candidates[:, start:])  # Add final segment
        return regions


def compute_upper_envelope(
    segments: List[np.ndarray], grid_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the upper envelope from overlapping value function segments."""

    # 1. Create a unified wealth grid
    unified_grid = np.unique(np.concatenate([seg[0] for seg in segments]))

    # 2. Interpolate all segments over the unified grid using NumPy
    # Since each segment itself is sorted, we can use np.interp
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

    # 4. Find unified gird points at which the best segment changes
    change_indx = np.where(best_segment[:-1] != best_segment[1:])[0]
    prev_seg = best_segment[change_indx]
    curr_seg = best_segment[change_indx + 1]

    # 4. Initialize kink array
    kinks = np.empty((3, len(change_indx) * 2))

    # 6. Iterate through the change indices to find kinks
    for insert_idx, change_idx, prev_seg, curr_seg in zip(
        range(0, len(change_indx) * 2, 2),
        change_indx,
        prev_seg,
        curr_seg,
    ):
        x_kink, value_kink, interp_left, interp_right = calculate_kink(
            change_idx,
            prev_seg,
            curr_seg,
            interpolated_value,
            interpolated_policy,
            unified_grid,
        )
        # Store kink endo_grid, value and policy left and right of the kink
        kinks[:, insert_idx : insert_idx + 2] = np.array(
            [
                [x_kink - EPS, x_kink + EPS],
                [value_kink, value_kink],
                [interp_left, interp_right],
            ]
        )

    # Remove kinks with NaN values
    kinks = kinks[:, np.all(np.isfinite(kinks), axis=0)]

    # Find the insert positions for all kinks in the unified grid
    insert_positions = np.searchsorted(unified_grid, kinks[0, :])

    # Create base policy array using best_segment
    base_policy = interpolated_policy[
        best_segment, np.arange(interpolated_policy.shape[1])
    ]

    # Insert kinks into the unified grid, values, and policy
    endog_grid = np.insert(unified_grid, insert_positions, kinks[0, :])
    value = np.insert(max_values, insert_positions, kinks[1, :])
    policy = np.insert(base_policy, insert_positions, kinks[2, :])

    # Pad with NaNs to match the final grid size
    padding_size = grid_size - endog_grid.size
    endog_grid = np.pad(
        endog_grid, (0, padding_size), mode="constant", constant_values=np.nan
    )
    policy = np.pad(policy, (0, padding_size), mode="constant", constant_values=np.nan)
    value = np.pad(value, (0, padding_size), mode="constant", constant_values=np.nan)

    return endog_grid, policy, value, kinks


def _augment_grid_left(
    candidates: np.ndarray,
    state_choice_dict: Dict,
    expected_value_zero_assets: float,
    min_wealth: float,
    avg_grid_step: float,
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
    # if state_choice_dict == {'choice': np.uint8(0), 'dummy_stochastic': np.uint8(0), 'lagged_choice': np.uint8(0), 'period': np.uint8(14)}:
    #     breakpoint()
    extra_points = np.arange(EPS, candidates[0, 0], avg_grid_step * 0.1)
    # remove last point if is greater or equal to min_wealth
    if extra_points.size == 0:
        return np.concat(
            (np.vstack((0, 0, expected_value_zero_assets)), candidates), axis=1
        )
    if extra_points[-1] >= min_wealth:
        extra_points = extra_points[:-1]
    utility = compute_utility(
        consumption=extra_points, params=params, **state_choice_dict
    )
    extra_values = utility + params["beta"] * expected_value_zero_assets

    return np.vstack(
        (
            np.concat(([0], extra_points, candidates[0, :])),
            np.concat(([0], extra_points, candidates[1, :])),
            np.concat(([expected_value_zero_assets], extra_values, candidates[2, :])),
        )
    )


def interpolate_with_extrapolation(x, y, x0):
    return y[0] + ((y[1] - y[0]) / (x[1] - x[0])) * (x0 - x[0])


def calculate_kink(
    change_indx: int,
    prev_seg: int,
    curr_seg: int,
    interpolated_value: np.ndarray,
    interpolated_policy: np.ndarray,
    unified_grid: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Calculate the kink point and its value and policy."""
    x0 = unified_grid[change_indx]
    x1 = unified_grid[change_indx + 1]
    y0_prev = interpolated_value[prev_seg, change_indx]
    y1_prev = interpolated_value[prev_seg, change_indx + 1]
    y0_curr = interpolated_value[curr_seg, change_indx]
    y1_curr = interpolated_value[curr_seg, change_indx + 1]

    p0_prev = interpolated_policy[prev_seg, change_indx]
    p1_prev = interpolated_policy[prev_seg, change_indx + 1]
    p0_curr = interpolated_policy[curr_seg, change_indx]
    p1_curr = interpolated_policy[curr_seg, change_indx + 1]

    if all(np.isfinite([y0_prev, y1_prev, y0_curr, y1_curr])):

        slope_prev = (y1_prev - y0_prev) / (x1 - x0)
        intercept_prev = y0_prev - slope_prev * x0
        slope_curr = (y1_curr - y0_curr) / (x1 - x0)
        intercept_curr = y0_curr - slope_curr * x0

        x_kink = (intercept_curr - intercept_prev) / (slope_prev - slope_curr)
        value_kink = slope_prev * x_kink + intercept_prev
        interp_left = interpolate_with_extrapolation(
            [x0, x1], [p0_prev, p1_prev], x_kink
        )
        interp_right = interpolate_with_extrapolation(
            [x0, x1], [p0_curr, p1_curr], x_kink
        )
        return x_kink, value_kink, interp_left, interp_right
    else:
        return np.nan, np.nan, np.nan, np.nan
