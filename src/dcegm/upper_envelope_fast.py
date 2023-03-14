"""Implementation of the Fast Upper-Envelope Scan.

Based on Akshay Shanker, University of Sydney, akshay.shanker@me.com.

"""
from functools import partial
from typing import Callable
from typing import Optional
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from dcegm.interpolate import linear_interpolation_with_extrapolation  # noqa: F401
from jax import jit


def fast_upper_envelope_wrapper(
    policy: np.ndarray,
    value: np.ndarray,
    exog_grid: np.ndarray,
    choice: int,  # noqa: U100
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
        exog_grid (np.ndarray): 1d array of exogenous savings grid of shape
            (n_grid_wealth,).
        choice (int): The current choice.
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
    n_grid_wealth = len(exog_grid)
    exog_grid = np.append(0, exog_grid)

    endog_grid = policy[0]
    policy_ = policy[1]
    value_ = value[1]

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

    # TODO: determine locations where endogenous grid points are # noqa: T000
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
    exog_grid = np.take(exog_grid, np.argsort(endog_grid))
    endog_grid = np.sort(endog_grid)

    # ================================================================================

    value_clean_with_nans, policy_clean_with_nans, endog_grid_clean_with_nans = _scan(
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


# ================================================================================


def _scan(
    endog_grid,
    value,
    policy,
    exog_grid,
    jump_thresh,
    n_points_to_scan=10,
):
    """Scan the value function to remove suboptimal points.

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
    # leading index for optimal values j
    # leading index for value to be `checked' is i+1

    value_refined = np.copy(value)
    policy_refined = np.copy(policy)
    endog_grid_refined = np.copy(endog_grid)

    value_refined[3:] = np.nan
    endog_grid_refined[3:] = np.nan
    policy_refined[3:] = np.nan

    suboptimal_points = np.zeros(n_points_to_scan, dtype=int)

    j = 1
    k = 0

    idx_refined = 3

    for i in range(2, len(endog_grid) - 2):
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

            # if right turn is made and jump registered
            # remove point or perform forward scan
            if grad_before > grad_next and switch_value_func:
                keep_next = False

                grad_next_forward, _, found_next_point_on_same_value = _forward_scan(
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
                    value_refined[idx_refined] = value[i + 1]
                    policy_refined[idx_refined] = policy[i + 1]
                    endog_grid_refined[idx_refined] = endog_grid[i + 1]
                    idx_refined += 1
                    k = j
                    j = i + 1

            elif grad_before > grad_next and exog_grid[i + 1] - exog_grid[j] < 0:
                suboptimal_points = _append_new_point(suboptimal_points, i + 1)

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

                idx_before_on_upper_curve = suboptimal_points[dist_before_on_same_value]

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

                if not keep_current:
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

                    # =================================================================

                    if idx_before_on_upper_curve > 0:
                        value_refined[idx_refined - 1] = intersect_value
                        policy_refined[idx_refined - 1] = intersect_policy_left
                        endog_grid_refined[idx_refined - 1] = intersect_grid

                        value_refined[idx_refined] = intersect_value
                        policy_refined[idx_refined] = intersect_policy_right
                        endog_grid_refined[idx_refined] = intersect_grid
                        idx_refined += 1

                    # =================================================================

                    value_refined[idx_refined] = value[i + 1]
                    policy_refined[idx_refined] = policy[i + 1]
                    endog_grid_refined[idx_refined] = endog_grid[i + 1]
                    idx_refined += 1

                    value[j] = intersect_value
                    policy[j] = intersect_policy_right
                    endog_grid[j] = intersect_grid

                    j = i + 1

                else:
                    # ================== NEW CODE ==================
                    if grad_next > grad_before and switch_value_func:
                        (
                            grad_next_forward,
                            dist_next_point_on_same_value,
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

                        idx_next_on_lower_curve = j + dist_next_point_on_same_value + 2

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

                    # ================== OLD CODE ==================

                    value_refined[idx_refined] = value[i + 1]
                    policy_refined[idx_refined] = policy[i + 1]
                    endog_grid_refined[idx_refined] = endog_grid[i + 1]
                    idx_refined += 1
                    k = j
                    j = i + 1

    # The last point is in there by definition?!
    value_refined[idx_refined] = value[-1]
    endog_grid_refined[idx_refined] = endog_grid[-1]
    policy_refined[idx_refined] = policy[-1]

    return value_refined, policy_refined, endog_grid_refined


@partial(jit, static_argnums=(6))
def _forward_scan(
    value,
    endog_grid,
    exog_grid,
    jump_thresh,
    idx_current,
    idx_next,
    n_points_to_scan,
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
    dist_next_on_same_value = 0
    grad_next_on_same_value = 0

    for i in range(1, n_points_to_scan + 1):
        is_on_same_value = (
            jnp.abs(
                (exog_grid[idx_current] - exog_grid[idx_next + i])
                / (endog_grid[idx_current] - endog_grid[idx_next + i])
            )
            < jump_thresh
        )
        is_next = is_on_same_value * (1 - is_next_on_same_value)
        dist_next_on_same_value = (i - 1) * is_next + (
            1 - is_next
        ) * dist_next_on_same_value

        grad_next_on_same_value = (
            (value[idx_next] - value[idx_next + i])
            / (endog_grid[idx_next] - endog_grid[idx_next + i])
        ) * is_next + (1 - is_next) * grad_next_on_same_value

        is_next_on_same_value = (
            is_next_on_same_value * is_on_same_value
            + (1 - is_on_same_value) * is_next_on_same_value
            + is_on_same_value * (1 - is_next_on_same_value)
        )

    return (
        grad_next_on_same_value,
        dist_next_on_same_value,
        is_next_on_same_value,
    )


@jit
def _backward_scan(
    value_unrefined,
    endog_grid,
    exog_grid,
    suboptimal_points,
    jump_thresh,
    idx_current,
    idx_next,
):
    """Scan backward to find the previous optimal point.

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

    for i, idx_before_on_same_value in enumerate(suboptimal_points[::-1]):
        is_on_same_value = (
            jnp.abs(
                (exog_grid[idx_next] - exog_grid[idx_before_on_same_value])
                / (endog_grid[idx_next] - endog_grid[idx_before_on_same_value])
            )
            < jump_thresh
        )
        is_before = is_on_same_value * (1 - is_before_on_same_value)
        dist_before_on_same_value = (indexes_reversed - i) * is_before + (
            1 - is_before
        ) * dist_before_on_same_value

        grad_before_on_same_value = (
            (value_unrefined[idx_current] - value_unrefined[idx_before_on_same_value])
            / (endog_grid[idx_current] - endog_grid[idx_before_on_same_value])
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


def find_intersection_point_grid_and_value(a1, a2, b1, b2):
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
    denom = dap @ db
    num = dap @ dp
    return tuple((num / denom) * db + b1)


def _append_new_point(x_array, m):
    """Append a new point to an array."""
    for i in range(len(x_array) - 1):
        x_array[i] = x_array[i + 1]

    x_array[-1] = m
    return x_array


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
