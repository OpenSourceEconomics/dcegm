"""Implementation of the Fast Upper-Envelope Scan.

Based on Akshay Shanker, University of Sydney, akshay.shanker@me.com.

"""
from typing import Callable
from typing import Optional
from typing import Tuple

import jax.numpy as jnp  # noqa: F401
import numpy as np
from dcegm.interpolate import linear_interpolation_with_extrapolation
from jax import jit  # noqa: F401


def fast_upper_envelope_wrapper(
    policy: np.ndarray,
    value: np.ndarray,
    exog_grid: np.ndarray,
    choice: int,  # noqa: U100
    compute_value: Callable,  # noqa: U100
    period,  # noqa: U100
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
        exog_grid (np.ndarray): 1d array of exogenous savings grid of shape
            (n_grid_wealth,).
        choice (int): The current choice.
        compute_value (callable): Function to compute the agent's value.
        period (int): The current period.

    Returns:
        tuple:

        - policy_refined_with_nans (np.ndarray): Worker's *refined* (consumption) policy
            function of the current period, where suboptimal points have been dropped.
            Shape (2, 1.1 * n_grid_wealth).
        - value_refined_with_nans (np.ndarray): Worker's *refined* value function of the
            current period, where suboptimal points have been dropped.
            Shape (2, 1.1 * n_grid_wealth).

    """
    endog_grid = np.copy(policy[0])
    value = np.copy(value[1])
    policy = np.copy(policy[1])
    n_grid_wealth = len(exog_grid)

    min_wealth_grid = np.min(endog_grid[1:])
    if endog_grid[1] > min_wealth_grid:
        # Non-concave region coincides with credit constraint.
        # This happens when there is a non-monotonicity in the endogenous wealth grid
        # that goes below the first point.
        # Solution: Value function to the left of the first point is analytical,
        # so we just need to add some points to the left of the first grid point.

        endog_grid_augmented, value_augmented, policy_augmented = _augment_grids(
            endog_grid=endog_grid,
            value=value,
            policy=policy,
            choice=choice,
            expected_value_zero_wealth=value[0],
            min_wealth_grid=min_wealth_grid,
            n_grid_wealth=n_grid_wealth,
            compute_value=compute_value,
        )
        endog_grid = np.append(0, endog_grid_augmented)
        policy = np.append(policy[0], policy_augmented)
        value = np.append(value[0], value_augmented)
        exog_grid_augmented = np.linspace(
            exog_grid[1], exog_grid[2], n_grid_wealth // 10 + 1
        )
        exog_grid = np.append([0], np.append(exog_grid_augmented, exog_grid[2:]))
    else:
        exog_grid = np.append(0, exog_grid)

    endog_grid_refined, value_refined, policy_refined = fast_upper_envelope(
        endog_grid, value, policy, exog_grid, jump_thresh=2
    )

    # Fill array with nans to fit 10% extra grid points
    policy_refined_with_nans = np.empty((2, int(1.1 * n_grid_wealth)))
    value_refined_with_nans = np.empty((2, int(1.1 * n_grid_wealth)))
    policy_refined_with_nans[:] = np.nan
    value_refined_with_nans[:] = np.nan

    policy_refined_with_nans[0, : policy_refined.shape[0]] = endog_grid_refined
    policy_refined_with_nans[1, : policy_refined.shape[0]] = policy_refined
    value_refined_with_nans[0, : value_refined.shape[0]] = endog_grid_refined
    value_refined_with_nans[1, : value_refined.shape[0]] = value_refined

    return (
        policy_refined_with_nans,
        value_refined_with_nans,
    )


def fast_upper_envelope(
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    exog_grid: np.ndarray,
    jump_thresh: Optional[float] = 2,
    lower_bound_wealth: Optional[float] = 1e-10,
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
        lower_bound_wealth (float): Lower bound on wealth.

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
    # TODO: determine locations where endogenous grid points are # noqa: T000
    # equal to the lower bound
    # mask = endog_grid <= lower_bound_wealth
    # if np.any(mask):
    #     max_value_lower_bound = np.nanmax(value[mask])
    #     mask &= value < max_value_lower_bound
    #     value[mask] = np.nan

    endog_grid = endog_grid[np.where(~np.isnan(value))]
    policy = policy[np.where(~np.isnan(value))]
    exog_grid = exog_grid[np.where(~np.isnan(value))]
    value = value[np.where(~np.isnan(value))]

    idx_sort = np.argsort(endog_grid, kind="mergesort")
    value = np.take(value, idx_sort)
    policy = np.take(policy, idx_sort)
    exog_grid = np.take(exog_grid, idx_sort)
    endog_grid = np.take(endog_grid, idx_sort)

    (
        value_clean_with_nans,
        policy_clean_with_nans,
        endog_grid_clean_with_nans,
    ) = scan_value_function(
        endog_grid,
        value,
        policy,
        exog_grid,
        jump_thresh=jump_thresh,
        n_points_to_scan=10,
    )

    endog_grid_refined = endog_grid_clean_with_nans[
        ~np.isnan(endog_grid_clean_with_nans)
    ]
    value_refined = value_clean_with_nans[~np.isnan(value_clean_with_nans)]
    policy_refined = policy_clean_with_nans[~np.isnan(policy_clean_with_nans)]

    return endog_grid_refined, value_refined, policy_refined


def scan_value_function(
    endog_grid,
    value,
    policy,
    exog_grid,
    jump_thresh,
    n_points_to_scan=10,
):
    """Scan the value function to remove suboptimal points and add itersection points.

    Args:
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth + 1,).
        polcy (np.ndarray): 1d array containing the unrefined policy correspondence
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
    policy_refined = np.copy(policy)
    endog_grid_refined = np.copy(endog_grid)

    value_refined[2:] = np.nan
    endog_grid_refined[2:] = np.nan
    policy_refined[2:] = np.nan

    suboptimal_points = np.zeros(n_points_to_scan, dtype=int)

    j = 1
    k = 0

    idx_refined = 2

    for i in range(1, len(endog_grid) - 2):
        if value[i + 1] - value[j] < 0:
            suboptimal_points = _append_new_point(suboptimal_points, i + 1)

        else:
            # value function gradient between previous two optimal points
            grad_before = (value[j] - value[k]) / (endog_grid[j] - endog_grid[k])

            # gradient with leading index to be checked
            grad_next = (value[i + 1] - value[j]) / (endog_grid[i + 1] - endog_grid[j])

            switch_value_func = (
                np.abs(
                    (exog_grid[i + 1] - exog_grid[j])
                    / (endog_grid[i + 1] - endog_grid[j])
                )
                > jump_thresh
            )

            if grad_before > grad_next and exog_grid[i + 1] - exog_grid[j] < 0:
                suboptimal_points = _append_new_point(suboptimal_points, i + 1)

            # if right turn is made and jump registered
            # remove point or perform forward scan
            elif grad_before > grad_next and switch_value_func:
                keep_next = False

                (
                    grad_next_forward,
                    idx_next_on_lower_curve,
                    found_next_point_on_same_value,
                ) = _forward_scan(
                    value=value,
                    endog_grid=endog_grid,
                    exog_grid=exog_grid,
                    jump_thresh=jump_thresh,
                    idx_current=j,
                    idx_next=i + 1,
                    n_points_to_scan=n_points_to_scan,
                )

                # get index of closest next point with same discrete choice as point j
                if found_next_point_on_same_value:
                    if grad_next > grad_next_forward:
                        keep_next = True

                if not keep_next:
                    suboptimal_points = _append_new_point(suboptimal_points, i + 1)
                else:
                    grad_next_backward, dist_before_on_same_value = _backward_scan(
                        value_unrefined=value,
                        endog_grid=endog_grid,
                        exog_grid=exog_grid,
                        suboptimal_points=suboptimal_points,
                        jump_thresh=jump_thresh,
                        idx_current=j,
                        idx_next=i + 1,
                    )
                    idx_before_on_upper_curve = suboptimal_points[
                        dist_before_on_same_value
                    ]

                    intersect_grid, intersect_value = _linear_intersection(
                        x1=endog_grid[idx_next_on_lower_curve],
                        y1=value[idx_next_on_lower_curve],
                        x2=endog_grid[j],
                        y2=value[j],
                        x3=endog_grid[i + 1],
                        y3=value[i + 1],
                        x4=endog_grid[idx_before_on_upper_curve],
                        y4=value[idx_before_on_upper_curve],
                    )

                    intersect_policy_left = _evaluate_point_on_line(
                        x1=endog_grid[idx_next_on_lower_curve],
                        y1=policy[idx_next_on_lower_curve],
                        x2=endog_grid[j],
                        y2=policy[j],
                        point_to_evaluate=intersect_grid,
                    )

                    intersect_policy_right = _evaluate_point_on_line(
                        x1=endog_grid[i + 1],
                        y1=policy[i + 1],
                        x2=endog_grid[idx_before_on_upper_curve],
                        y2=policy[idx_before_on_upper_curve],
                        point_to_evaluate=intersect_grid,
                    )

                    value_refined[idx_refined] = intersect_value
                    policy_refined[idx_refined] = intersect_policy_left
                    endog_grid_refined[idx_refined] = intersect_grid
                    idx_refined += 1

                    value_refined[idx_refined] = intersect_value
                    policy_refined[idx_refined] = intersect_policy_right
                    endog_grid_refined[idx_refined] = intersect_grid
                    idx_refined += 1

                    value_refined[idx_refined] = value[i + 1]
                    policy_refined[idx_refined] = policy[i + 1]
                    endog_grid_refined[idx_refined] = endog_grid[i + 1]
                    idx_refined += 1
                    k = j
                    j = i + 1

            # if left turn is made or right turn with no jump, then
            # keep point provisionally and conduct backward scan
            else:
                grad_next_backward, dist_before_on_same_value = _backward_scan(
                    value_unrefined=value,
                    endog_grid=endog_grid,
                    exog_grid=exog_grid,
                    suboptimal_points=suboptimal_points,
                    jump_thresh=jump_thresh,
                    idx_current=j,
                    idx_next=i + 1,
                )
                keep_current = True
                current_is_optimal = True
                idx_before_on_upper_curve = suboptimal_points[dist_before_on_same_value]

                # # This should better a bool from the backwards scan
                grad_next_forward, *_ = _forward_scan(
                    value=value,
                    endog_grid=endog_grid,
                    exog_grid=exog_grid,
                    jump_thresh=jump_thresh,
                    idx_current=j,
                    idx_next=i + 1,
                    n_points_to_scan=n_points_to_scan,
                )
                if grad_next_forward > grad_next and switch_value_func:
                    suboptimal_points = _append_new_point(suboptimal_points, i + 1)
                    current_is_optimal = False

                # if the gradient joining the leading point i+1 (we have just
                # jumped to) and the point m(the last point on the same
                # choice specific policy) is shallower than the
                # gradient joining the i+1 and j, then delete j'th point
                if (
                    grad_before < grad_next
                    and grad_next >= grad_next_backward
                    and switch_value_func
                ):
                    keep_current = False

                if not keep_current and current_is_optimal:
                    intersect_grid, intersect_value = _linear_intersection(
                        x1=endog_grid[j],
                        y1=value[j],
                        x2=endog_grid[k],
                        y2=value[k],
                        x3=endog_grid[i + 1],
                        y3=value[i + 1],
                        x4=endog_grid[idx_before_on_upper_curve],
                        y4=value[idx_before_on_upper_curve],
                    )

                    # The next two interpolations is just to show that from
                    # interpolation from each side leads to the same result
                    intersect_policy_left = linear_interpolation_with_extrapolation(
                        x=np.array([endog_grid[j], endog_grid[k]]),
                        y=np.array([policy[j], policy[k]]),
                        x_new=intersect_grid,
                    )
                    intersect_policy_right = linear_interpolation_with_extrapolation(
                        x=np.array(
                            [endog_grid[i + 1], endog_grid[idx_before_on_upper_curve]]
                        ),
                        y=np.array([policy[i + 1], policy[idx_before_on_upper_curve]]),
                        x_new=intersect_grid,
                    )

                    if idx_before_on_upper_curve > 0 and i > 1:
                        value_refined[idx_refined - 1] = intersect_value
                        policy_refined[idx_refined - 1] = intersect_policy_left
                        endog_grid_refined[idx_refined - 1] = intersect_grid

                        value_refined[idx_refined] = intersect_value
                        policy_refined[idx_refined] = intersect_policy_right
                        endog_grid_refined[idx_refined] = intersect_grid
                        idx_refined += 1

                    value_refined[idx_refined] = value[i + 1]
                    policy_refined[idx_refined] = policy[i + 1]
                    endog_grid_refined[idx_refined] = endog_grid[i + 1]
                    idx_refined += 1

                    value[j] = intersect_value
                    policy[j] = intersect_policy_right
                    endog_grid[j] = intersect_grid

                    j = i + 1

                elif keep_current and current_is_optimal:
                    if grad_next > grad_before and switch_value_func:
                        (
                            grad_next_forward,
                            idx_next_on_lower_curve,
                            _,
                        ) = _forward_scan(
                            value=value,
                            endog_grid=endog_grid,
                            exog_grid=exog_grid,
                            jump_thresh=jump_thresh,
                            idx_current=j,
                            idx_next=i + 1,
                            n_points_to_scan=n_points_to_scan,
                        )

                        intersect_grid, intersect_value = _linear_intersection(
                            x1=endog_grid[idx_next_on_lower_curve],
                            y1=value[idx_next_on_lower_curve],
                            x2=endog_grid[j],
                            y2=value[j],
                            x3=endog_grid[i + 1],
                            y3=value[i + 1],
                            x4=endog_grid[idx_before_on_upper_curve],
                            y4=value[idx_before_on_upper_curve],
                        )

                        intersect_policy_left = linear_interpolation_with_extrapolation(
                            x=np.array(
                                [endog_grid[idx_next_on_lower_curve], endog_grid[j]]
                            ),
                            y=np.array([policy[idx_next_on_lower_curve], policy[j]]),
                            x_new=intersect_grid,
                        )
                        intersect_policy_right = (
                            linear_interpolation_with_extrapolation(
                                x=np.array(
                                    [
                                        endog_grid[i + 1],
                                        endog_grid[idx_before_on_upper_curve],
                                    ]
                                ),
                                y=np.array(
                                    [policy[i + 1], policy[idx_before_on_upper_curve]]
                                ),
                                x_new=intersect_grid,
                            )
                        )

                        value_refined[idx_refined] = intersect_value
                        policy_refined[idx_refined] = intersect_policy_left
                        endog_grid_refined[idx_refined] = intersect_grid
                        idx_refined += 1

                        value_refined[idx_refined] = intersect_value
                        policy_refined[idx_refined] = intersect_policy_right
                        endog_grid_refined[idx_refined] = intersect_grid
                        idx_refined += 1

                    value_refined[idx_refined] = value[i + 1]
                    policy_refined[idx_refined] = policy[i + 1]
                    endog_grid_refined[idx_refined] = endog_grid[i + 1]
                    idx_refined += 1
                    k = j
                    j = i + 1

    value_refined[idx_refined] = value[-1]
    endog_grid_refined[idx_refined] = endog_grid[-1]
    policy_refined[idx_refined] = policy[-1]

    return value_refined, policy_refined, endog_grid_refined


def _forward_scan(
    value,
    endog_grid,
    exog_grid,
    jump_thresh,
    idx_current,
    idx_next,
    n_points_to_scan,
):
    """Scan forward to check whether next point is optimal.

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
        n_points_to_scan (int): The number of points to scan forward.
        index_on_curve_to_pick (int): The index on the curve to pick.

    Returns:
        tuple:

        - grad_next_forward (float): The gradient of the next point on the same
            value function.
        - is_point_on_same_value (int): Indicator for whether the next point is on
            the same value function.
        - dist_next_point_on_same_value (int): The distance to the next point on
            the same value function.

    """

    is_next_on_same_value = 0
    idx_on_same_value = 0
    grad_next_on_same_value = 0

    idx_max = exog_grid.shape[0] - 1

    for i in range(1, n_points_to_scan + 1):
        idx_to_check = min(idx_next + i, idx_max)
        if endog_grid[idx_current] < endog_grid[idx_to_check]:
            is_on_same_value = (
                np.abs(
                    (exog_grid[idx_current] - exog_grid[idx_to_check])
                    / (endog_grid[idx_current] - endog_grid[idx_to_check])
                )
                < jump_thresh
            )
            is_next = is_on_same_value * (1 - is_next_on_same_value)
            idx_on_same_value = (
                idx_to_check * is_next + (1 - is_next) * idx_on_same_value
            )

            grad_next_on_same_value = (
                (value[idx_next] - value[idx_to_check])
                / (endog_grid[idx_next] - endog_grid[idx_to_check])
            ) * is_next + (1 - is_next) * grad_next_on_same_value

            is_next_on_same_value = (
                is_next_on_same_value * is_on_same_value
                + (1 - is_on_same_value) * is_next_on_same_value
                + is_on_same_value * (1 - is_next_on_same_value)
            )

    return (
        grad_next_on_same_value,
        idx_on_same_value,
        is_next_on_same_value,
    )


def _backward_scan(
    value_unrefined,
    endog_grid,
    exog_grid,
    suboptimal_points,
    jump_thresh,
    idx_current,
    idx_next,
):
    """Scan backward to check whether current point is optimal.

    Args:
        value (np.ndarray): 1d array containing the value function of shape
            (n_grid_wealth + 1,).
        endog_grid (np.ndarray): 1d array containing the endogenous wealth grid of
            shape (n_grid_wealth + 1,).
        exog_grid (np.ndarray): 1d array containing the exogenous wealth grid of
            shape (n_grid_wealth + 1,).
        suboptimal_points (list): List of suboptimal points in the value functions.
        jump_thresh (float): Threshold for the jump in the value function.
        idx_current (int): Index of the current point in the value function.
        idx_next (int): Index of the next point in the value function.

    Returns:
        tuple:

        - grad_before_on_same_value (float): The gradient of the previous point on
            the same value function.
        - is_before_on_same_value (int): Indicator for whether we have found a
            previous point on the same value function.

    """

    is_before_on_same_value = 0
    dist_before_on_same_value = 0
    grad_before_on_same_value = 0

    indexes_reversed = len(suboptimal_points) - 1

    for i, idx_to_check in enumerate(suboptimal_points[::-1]):
        if endog_grid[idx_current] > endog_grid[idx_to_check]:
            is_on_same_value = (
                np.abs(
                    (exog_grid[idx_next] - exog_grid[idx_to_check])
                    / (endog_grid[idx_next] - endog_grid[idx_to_check])
                )
                < jump_thresh
            )
            is_before = is_on_same_value * (1 - is_before_on_same_value)
            dist_before_on_same_value = (indexes_reversed - i) * is_before + (
                1 - is_before
            ) * dist_before_on_same_value

            grad_before_on_same_value = (
                (value_unrefined[idx_current] - value_unrefined[idx_to_check])
                / (endog_grid[idx_current] - endog_grid[idx_to_check])
            ) * is_before + (1 - is_before) * grad_before_on_same_value

            is_before_on_same_value = (
                (is_before_on_same_value * is_on_same_value)
                + (1 - is_on_same_value) * is_before_on_same_value
                + is_on_same_value * (1 - is_before_on_same_value)
            )

    return (
        grad_before_on_same_value,
        dist_before_on_same_value,
    )


def _evaluate_point_on_line(x1, y1, x2, y2, point_to_evaluate):
    """Evaluate a point on a line.

    Args:
        x1 (float): x coordinate of the first point.
        y1 (float): y coordinate of the first point.
        x2 (float): x coordinate of the second point.
        y2 (float): y coordinate of the second point.
        point_to_evaluate (float): The point to evaluate.

    Returns:
        float: The value of the point on the line.

    """
    return (y2 - y1) / (x2 - x1) * (point_to_evaluate - x1) + y1


def _linear_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """Find the intersection of two lines.

    Args:

        x1 (float): x-coordinate of the first point of the first line.
        y1 (float): y-coordinate of the first point of the first line.
        x2 (float): x-coordinate of the second point of the first line.
        y2 (float): y-coordinate of the second point of the first line.
        x3 (float): x-coordinate of the first point of the second line.
        y3 (float): y-coordinate of the first point of the second line.
        x4 (float): x-coordinate of the second point of the second line.
        y4 (float): y-coordinate of the second point of the second line.

    Returns:
        tuple: x and y coordinates of the intersection point.

    """

    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (y4 - y3) / (x4 - x3)

    x_intersection = (slope1 * x1 - slope2 * x3 + y3 - y1) / (slope1 - slope2)
    y_intersection = slope1 * (x_intersection - x1) + y1

    return x_intersection, y_intersection


def _augment_grids(
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    choice: int,
    expected_value_zero_wealth: np.ndarray,
    min_wealth_grid: float,
    n_grid_wealth: int,
    compute_value: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extends the endogenous wealth grid, value, and policy functions to the left.

    Args:
        endog_grid (np.ndarray): 1d array containing the endogenous wealth grid of
            shape (n_endog_wealth_grid,), where n_endog_wealth_grid is of variable
            length depending on the number of kinks and non-concave regions in the
            value function.
        value (np.ndarray):  1d array storing the choice-specific
            value function of shape (n_endog_wealth_grid,), where
            n_endog_wealth_grid is of variable length depending on the number of
            kinks and non-concave regions in the value function.
            In the presence of kinks, the value function is a "correspondence"
            rather than a function due to non-concavities.
        policy (np.ndarray):  1d array storing the choice-specific
            policy function of shape (n_endog_wealth_grid,), where
            n_endog_wealth_grid is of variable length depending on the number of
            discontinuities in the policy function.
            In the presence of discontinuities, the policy function is a
            "correspondence" rather than a function due to multiple local optima.
        choice (int): The agent's choice.
        expected_value_zero_wealth (float): The agent's expected value given that she
            has a wealth of zero.
        min_wealth_grid (float): Minimal wealth level in the endogenous wealth grid.
        n_grid_wealth (int): Number of grid points in the exogenous wealth grid.
        compute_value (callable): Function to compute the agent's value.

    Returns:
        tuple:

        - grid_augmented (np.ndarray): 1d array containing the augmented
            endogenous wealth grid with ancillary points added to the left.
        - policy_augmented (np.ndarray): 1d array containing the augmented
            policy function with ancillary points added to the left.
        - value_augmented (np.ndarray): 1d array containing the augmented
            value function with ancillary points added to the left.

    """
    grid_points_to_add = np.linspace(
        min_wealth_grid, endog_grid[1], n_grid_wealth // 10
    )[:-1]

    endog_grid_augmented = np.append(grid_points_to_add, endog_grid[1:])
    values_to_add = compute_value(
        grid_points_to_add,
        expected_value_zero_wealth,
        choice,
    )
    value_augmented = np.append(values_to_add, value[1:])
    policy_augmented = np.append(grid_points_to_add, policy[1:])

    return endog_grid_augmented, value_augmented, policy_augmented


def _append_new_point(x_array, m):
    """Append a new point to an array."""
    for i in range(len(x_array) - 1):
        x_array[i] = x_array[i + 1]

    x_array[-1] = m
    return x_array
