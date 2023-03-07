"""Implementation of the Fast Upper-Envelope Scan.

Based on Akshay Shanker, University of Sydney, akshay.shanker@me.com.

"""
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np


def fast_upper_envelope_wrapper(
    policy: np.ndarray,
    value: np.ndarray,
    exog_grid: np.ndarray,
    choice: int,  # noqa: U100
    n_grid_wealth: int,
    compute_value: Callable,  # noqa: U100
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop suboptimal points and refine the endogenous grid, policy, and value.

    Computes the upper envelope over the overlapping segments of the
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
        policy (np.ndarray): Array of choice-specific consumption policy
            of shape (2, n_grid_wealth).
            Position [0, :] of the arrays contain the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the (consumption) policy
            function c(M, d), for each time period and each discrete choice.
        value (np.ndarray): Array of choice-specific value function
            of shape (2, n_grid_wealth).
            Position [0, :] of the array contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice.
        choice (int): The current choice.
        n_grid_wealth (int): Number of grid points in the exogenous wealth grid.
        compute_value (callable): Function to compute the agent's value.

    Returns:
        tuple:

        - policy_refined (np.ndarray): Worker's *refined* (consumption) policy
            function of the current period, where suboptimal points have been dropped.
            Shape (2, 1.1 * n_grid_wealth).
        - value_refined (np.ndarray): Worker's *refined* value function of the
            current period, where suboptimal points have been dropped.
            Shape (2, 1.1 * n_grid_wealth).

    """
    endog_grid = policy[0]
    policy_ = policy[1]
    value_ = value[1]
    exog_grid = np.append(0, exog_grid)

    endog_grid_refined, value_out, policy_out = fast_upper_envelope(
        endog_grid, value_, policy_, exog_grid, jump_thresh=2
    )

    # ================================================================================

    policy_removed = np.row_stack([endog_grid_refined, policy_out])
    value_removed = np.row_stack([endog_grid_refined, value_out])

    policy_refined = policy_removed
    value_refined = value_removed

    # Fill array with nans to fit 10% extra grid points
    policy_refined_with_nans = np.empty((2, int(1.1 * n_grid_wealth)))
    value_refined_with_nans = np.empty((2, int(1.1 * n_grid_wealth)))
    policy_refined_with_nans[:] = np.nan
    value_refined_with_nans[:] = np.nan

    policy_refined_with_nans[:, : policy_refined.shape[1]] = policy_refined
    value_refined_with_nans[:, : value_refined.shape[1]] = value_refined

    # ================================================================================

    return policy_refined_with_nans, value_refined_with_nans


def fast_upper_envelope(
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    exog_grid: np.ndarray,
    jump_thresh: Optional[float] = 2,
    b: Optional[float] = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove suboptimal points from the endogenous grid, policy, and value function.

    Args:
        endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
            grid of shape (n_grid_wealth + 1,).
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth + 1,).
        policy (np.ndarray): 1d array containing the unrefined policy correspondence
            of shape (n_grid_wealth + 1,).
        exog_grid (np.ndarray): 1d array containing the exogenous wealth grid
            of shape (n_grid_wealth + 1,).
        jump_thresh (float): Jump detection threshold.

    Returns:
        tuple:

        - endog_grid_refined (np.ndarray): 1d array containing the refined endogenous
            wealth grid of shape (n_grid_clean,), which maps only to the optimal points
            in the value function.
        - value_refined (np.ndarray): 1d array containing the refined value function
            of shape (n_grid_clean,). Overlapping segments have been removed and only
            the optimal points are kept.
        - policy_refined (np.ndarray): 1d array containing the refined policy function
            of shape (n_grid_clean,). Overlapping segments have been removed and only
            the optimal points are kept.

    """

    # To-Do: determine locations where enogenous grid points are
    # equal to the lower bound
    mask = endog_grid <= b
    if np.any(mask):
        max_value_lower_bound = np.nanmax(value[mask])
        mask &= value < max_value_lower_bound
        value[mask] = np.nan

    endog_grid = endog_grid[np.where(~np.isnan(value))]
    policy = policy[np.where(~np.isnan(value))]
    exog_grid = exog_grid[np.where(~np.isnan(value))]
    value = value[np.where(~np.isnan(value))]

    value = np.take(value, np.argsort(endog_grid))
    policy = np.take(policy, np.argsort(endog_grid))
    exog_grid = np.take(exog_grid, np.argsort(endog_grid[~np.isnan(endog_grid)]))
    endog_grid = np.sort(endog_grid)

    # ================================================================================

    value_clean_with_nans = scan_value_correspondence(
        value, endog_grid, exog_grid, jump_thresh=jump_thresh
    )

    endog_grid_refined = (endog_grid[np.where(~np.isnan(value_clean_with_nans))],)
    value_refined = (value_clean_with_nans[np.where(~np.isnan(value_clean_with_nans))],)
    policy_refined = (policy[np.where(~np.isnan(value_clean_with_nans))],)

    return endog_grid_refined, value_refined, policy_refined


def scan_value_correspondence(
    value, endog_grid, exog_grid, jump_thresh=2, n_points_to_scan=10
):
    """Scan the value function to remove suboptimal points.

    Args:
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth + 1,).
        endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
            grid of shape (n_grid_wealth + 1,).
        exog_grid (np.ndarray): 1d array containing the exogenous wealth grid
            of shape (n_grid_wealth + 1,).
        jump_thresh (float): Jump detection threshold.
        n_points_to_scan (int): Number of points to scan for suboptimal points.

    Returns:
        value_refined (np.ndarray): 1d array containing the refined value function
            of shape (n_grid_clean,). Overlapping segments have been removed and only
            the optimal points are kept.

    """
    value_refined = np.copy(value)
    suboptimal_points = np.zeros(n_points_to_scan)

    j = 1
    k = 0

    for i in range(2, len(endog_grid) - 2):

        grad_current = (value[j] - value[k]) / (endog_grid[j] - endog_grid[k])
        grad_next = (value[i + 1] - value[j]) / (endog_grid[i + 1] - endog_grid[j])
        switch_value_func = (
            np.abs(
                (exog_grid[i + 1] - exog_grid[j]) / (endog_grid[i + 1] - endog_grid[j])
            )
            > jump_thresh
        )

        # if right turn is made and jump registered
        # remove point or perform forward scan
        if grad_next <= grad_current and switch_value_func:
            keep_next_point = False
            grad_next_arr, switch_value_func_forward = _forward_scan(
                value,
                endog_grid,
                exog_grid,
                jump_thresh=jump_thresh,
                idx_current=j,
                idx_next=i + 1,
                n_points_to_scan=10,
            )

            if np.sum(switch_value_func_forward) > 0:
                idx_next_on_value = np.where(switch_value_func_forward)[0][0]
                grad_next_forward = grad_next_arr[idx_next_on_value]

                if grad_next > grad_next_forward:
                    keep_next_point = True

            if not keep_next_point:
                value_refined[i + 1] = np.nan
                suboptimal_points = _append_new_point(suboptimal_points, i + 1)
            else:
                k = j
                j = i + 1

        elif value[i + 1] - value[j] < 0:
            value_refined[i + 1] = np.nan
            suboptimal_points = _append_new_point(suboptimal_points, i + 1)

        elif grad_next < grad_current and exog_grid[i + 1] - exog_grid[j] < 0:
            value_refined[i + 1] = np.nan
            suboptimal_points = _append_new_point(suboptimal_points, i + 1)

        # if left turn is made or right turn with no jump, then
        # keep point provisionally and conduct backward scan
        else:
            grad_before_arr, switch_value_func_backward = _backward_scan(
                value,
                endog_grid,
                exog_grid,
                suboptimal_points=suboptimal_points,
                jump_thresh=jump_thresh,
                idx_current=j,
                idx_next=i + 1,
            )
            keep_current_point = True

            if np.sum(switch_value_func_backward) > 0:
                idx_before_on_value = np.where(switch_value_func_backward)[0][-1]
                g_m_vf = grad_before_arr[idx_before_on_value]
            else:
                idx_before_on_value = 0
                keep_current_point = True

            idx_suboptimal = int(suboptimal_points[idx_before_on_value])

            if (
                grad_next > grad_current
                and grad_current >= g_m_vf
                and switch_value_func
            ):
                keep_current_point = False

            if not keep_current_point:
                pj = np.array([endog_grid[j], value[j]])
                pi1 = np.array([endog_grid[i + 1], value[i + 1]])
                pk = np.array([endog_grid[k], value[k]])
                pm = np.array([endog_grid[idx_suboptimal], value[idx_suboptimal]])
                intrsect = seg_intersect(pj, pk, pi1, pm)

                value_refined[j] = np.nan
                value[j] = intrsect[1]
                endog_grid[j] = intrsect[0]
                j = i + 1
            else:
                k = j
                j = i + 1

    return value_refined


def _forward_scan(
    value, endog_grid, exog_grid, jump_thresh, idx_current, idx_next, n_points_to_scan
):
    """Scan forward to find the next optimal point.

    Args:
        value (np.ndarray): 1d array containing the value function of shape
            (n_grid_wealth + 1,).
        endog_grid (np.ndarray): 1d array containing the endogenous wealth grid of
            shape (n_grid_wealth + 1,).
        exog_grid (np.ndarray): 1d array containing the exogenous wealth grid of
            shape (n_grid_wealth + 1,).
        jump_thresh (float): Threshold for the jump in the value function.
        idx_current (int): Index of the current point in the value function.
        idx_next (int): Index of the next point in the value function.

    Returns:
        tuple:

        - grad_next_arr (np.ndarray): 1d array containing the gradient of the
            value function at the next point.
        - switch_value_func (np.ndarray): 1d array of booleans denoting whether we
            switch value functions at the corresponding points.

    """
    grad_next_arr = np.empty(n_points_to_scan)
    switch_value_func = np.empty(n_points_to_scan)

    for i in range(1, n_points_to_scan + 1):
        grad_next_arr[i - 1] = (value[idx_next] - value[idx_next + i]) / (
            endog_grid[idx_next] - endog_grid[idx_next + 1 + i]
        )
        switch_value_func[i - 1] = (
            np.abs(
                (exog_grid[idx_current] - exog_grid[idx_next + i])
                / (endog_grid[idx_current] - endog_grid[idx_next + i])
            )
            > jump_thresh
        )

    return grad_next_arr, switch_value_func


def _backward_scan(
    value, endog_grid, exog_grid, suboptimal_points, jump_thresh, idx_current, idx_next
):
    """Scan backward to find the previous optimal point.

    Args:
        value (np.ndarray): 1d array containing the value function of shape
            (n_grid_wealth + 1,).
        endog_grid (np.ndarray): 1d array containing the endogenous wealth grid of
            shape (n_grid_wealth + 1,).
        exog_grid (np.ndarray): 1d array containing the exogenous wealth grid of
            shape (n_grid_wealth + 1,).
        suboptimal_points (list): List of suboptimal points in the value function.
        jump_thresh (float): Threshold for the jump in the value function.
        idx_current (int): Index of the current point in the value function.
        idx_next (int): Index of the next point in the value function.

    Returns:
        tuple:

        - grad_arr_next (np.ndarray): 1d array containing the gradients of the next
            points in the value function.
        - switch_value_func (np.ndarray): 1d array of booleans denoting whether we
            switch value functions at the corresponding points.

    """
    grad_arr_next = np.empty(len(suboptimal_points))
    switch_value_func = np.empty(len(suboptimal_points))

    for m in range(len(switch_value_func)):
        m_int = int(suboptimal_points[m])
        grad_arr_next[m] = (value[idx_current] - value[m_int]) / (
            endog_grid[idx_current] - endog_grid[m_int]
        )
        switch_value_func[m] = (
            np.abs(
                (exog_grid[idx_next] - exog_grid[m_int])
                / (endog_grid[idx_next] - endog_grid[m_int])
            )
            > jump_thresh
        )

    return grad_arr_next, switch_value_func


def seg_intersect(a1, a2, b1, b2):
    """Find the intersection of two lines.

    Args:
        a1 (np.ndarray): 1d array containing the first point of the first line.
        a2 (np.ndarray): 1d array containing the second point of the first line.
        b1 (np.ndarray): 1d array containing the first point of the second line.
        b2 (np.ndarray): 1d array containing the second point of the second line.

    Returns:
        np.ndarray: 1d array containing the intersection point of the two lines.

    """
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = np.array([-da[1], da[0]])
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom) * db + b1


def _append_new_point(x_array, m):
    """Append a new point to an array."""
    for i in range(len(x_array) - 1):
        x_array[i] = x_array[i + 1]

    x_array[-1] = m
    return x_array
