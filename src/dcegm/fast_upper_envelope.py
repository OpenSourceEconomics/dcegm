"""Extension of the Fast Upper-Envelope Scan.

The original algorithm is based on Loretti I. Dobrescu and Akshay Shanker (2022) 'Fast
Upper-Envelope Scan for Solving Dynamic Optimization Problems',
https://dx.doi.org/10.2139/ssrn.4181302

"""
from typing import Callable
from typing import Optional
from typing import Tuple

import jax.numpy as jnp  # noqa: F401
import numpy as np
from jax import jit  # noqa: F401
from numba import njit


def fast_upper_envelope_wrapper(
    endog_grid: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    exog_grid: np.ndarray,
    expected_value_zero_savings: float,
    choice: int,
    compute_value: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific endogenous grid.
        policy (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific policy function.
        value (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific value function.
        exog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) of the
            exogenous savings grid.
        expected_value_zero_savings (float): The agent's expected value given that she
            saves zero.
        choice (int): The current choice.
        compute_value (callable): Function to compute the agent's value.

    Returns:
        tuple:

        - endog_grid_refined (np.ndarray): 1d array of shape (1.1 * n_grid_wealth,)
            containing the refined state- and choice-specific endogenous grid.
        - policy_refined_with_nans (np.ndarray): 1d array of shape (1.1 * n_grid_wealth)
            containing refined state- and choice-specificconsumption policy.
        - value_refined_with_nans (np.ndarray): 1d array of shape (1.1 * n_grid_wealth)
            containing refined state- and choice-specific value function.

    """
    min_wealth_grid = np.min(endog_grid)
    if endog_grid[0] > min_wealth_grid:
        # Non-concave region coincides with credit constraint.
        # This happens when there is a non-monotonicity in the endogenous wealth grid
        # that goes below the first point.
        # Solution: Value function to the left of the first point is analytical,
        # so we just need to add some points to the left of the first grid point.

        endog_grid, value, policy = _augment_grids(
            endog_grid=endog_grid,
            value=value,
            policy=policy,
            choice=choice,
            expected_value_zero_savings=expected_value_zero_savings,
            min_wealth_grid=min_wealth_grid,
            points_to_add=len(endog_grid) // 10,
            compute_value=compute_value,
        )

    endog_grid = np.append(0, endog_grid)
    policy = np.append(0, policy)
    value = np.append(expected_value_zero_savings, value)

    (
        endog_grid_refined,
        value_refined,
        policy_left_refined,
        policy_right_refined,
    ) = fast_upper_envelope(endog_grid, value, policy, jump_thresh=2)

    return (
        endog_grid_refined,
        policy_left_refined,
        policy_right_refined,
        value_refined,
    )


def fast_upper_envelope(
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    jump_thresh: Optional[float] = 2,
    lower_bound_wealth: Optional[float] = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    # Comment by Akshay: Determine locations where endogenous grid points are
    # equal to the lower bound. Not relevant for us.
    # mask = endog_grid <= lower_bound_wealth
    # if np.any(mask):
    #     max_value_lower_bound = np.nanmax(value[mask])
    #     mask &= value < max_value_lower_bound
    #     value[mask] = np.nan

    idx_sort = np.argsort(endog_grid, kind="mergesort")
    value = np.take(value, idx_sort)
    policy = np.take(policy, idx_sort)
    endog_grid = np.take(endog_grid, idx_sort)

    (
        value_refined,
        policy_left_refined,
        policy_right_refined,
        endog_grid_refined,
    ) = scan_value_function(
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        jump_thresh=jump_thresh,
        n_points_to_scan=10,
    )

    return endog_grid_refined, value_refined, policy_left_refined, policy_right_refined


def scan_value_function(
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    jump_thresh: float,
    n_points_to_scan: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan the value function to remove suboptimal points and add intersection points.

    Args:
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth + 1,).
        policy (np.ndarray): 1d array containing the unrefined policy correspondence
            of shape (n_grid_wealth + 1,).
        endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
            grid of shape (n_grid_wealth + 1,).
        jump_thresh (float): Jump detection threshold.
        n_points_to_scan (int): Number of points to scan for suboptimal points.

    Returns:
        tuple:

        - (np.ndarray): 1d array of shape (n_grid_clean,) containing the refined
            value function. Overlapping segments have been removed and only
            the optimal points are kept.

    """
    endog_grid - policy
    (
        value_refined,
        policy_left_refined,
        policy_right_refined,
        endog_grid_refined,
    ) = _initialize_refined_arrays(value, policy, endog_grid)
    # j = 1
    # k = 0

    possible_values_case_2 = (value[-1], np.nan)
    possible_policies_case_2 = (policy[-1], np.nan)
    possible_endog_grid_case_2 = (endog_grid[-1], np.nan)

    value_k_and_j = value[0], value[1]
    endog_grid_k_and_j = endog_grid[0], endog_grid[1]
    policy_k_and_j = policy[0], policy[1]

    idx_to_inspect = 1
    last_point_intersect = 0
    value_to_be_saved_next = value[0]
    policy_left_to_be_saved_next = policy[0]
    policy_right_to_be_saved_next = policy[0]
    endog_grid_to_be_saved_next = endog_grid[0]

    idx_case_2 = 0
    for idx_refined in range(len(value_refined)):
        is_this_the_last_point = idx_to_inspect == len(endog_grid) - 1
        # In each iteration we calculate the gradient of the value function
        grad_before_denominator = endog_grid_k_and_j[1] - endog_grid_k_and_j[0] + 1e-16
        grad_before = (value_k_and_j[1] - value_k_and_j[0]) / grad_before_denominator

        # gradient with leading index to be checked
        grad_next_denominator = (
            endog_grid[idx_to_inspect] - endog_grid_k_and_j[1] + 1e-16
        )
        grad_next = (value[idx_to_inspect] - value_k_and_j[1]) / grad_next_denominator

        switch_value_denominator = (
            endog_grid[idx_to_inspect] - endog_grid_k_and_j[1] + 1e-16
        )
        exog_grid_j = endog_grid_k_and_j[1] - policy_k_and_j[1]
        exog_grid_idx_to_inspect = endog_grid[idx_to_inspect] - policy[idx_to_inspect]
        switch_value_func = (
            np.abs((exog_grid_idx_to_inspect - exog_grid_j) / switch_value_denominator)
            > jump_thresh
        )

        (
            grad_next_forward,
            idx_next_on_lower_curve,
        ) = _forward_scan(
            value=value,
            endog_grid=endog_grid,
            policy=policy,
            jump_thresh=jump_thresh,
            endog_grid_current=endog_grid_k_and_j[1],
            exog_grid_current=exog_grid_j,
            idx_base=idx_to_inspect,
            n_points_to_scan=n_points_to_scan,
        )

        (
            grad_next_backward,
            idx_before_on_upper_curve,
        ) = _backward_scan(
            value=value,
            endog_grid=endog_grid,
            policy=policy,
            jump_thresh=jump_thresh,
            value_current=value_k_and_j[1],
            endog_grid_current=endog_grid_k_and_j[1],
            idx_base=idx_to_inspect,
            n_points_to_scan=n_points_to_scan,
        )

        # Check for suboptimality. This is either with decreasing value function, the
        # value function not montone in consumption or
        # if the gradient joining the leading point i+1 and the point j (the last point
        # on the same choice specific policy) is shallower than the
        # gradient joining the i+1 and j, then delete j'th point
        # If the point is the same as point j, this is always false and
        # switch_value_func as well. Therefore, the third if is chosen.
        suboptimal_cond = (
            value[idx_to_inspect] < value_k_and_j[1]
            or exog_grid_idx_to_inspect < exog_grid_j
            or (grad_next < grad_next_forward and switch_value_func)
        )

        next_point_past_intersect = (
            grad_before > grad_next or grad_next < grad_next_backward
        )
        point_j_past_intersect = grad_next > grad_next_backward

        # Generate cases. They are exclusive in ascending order, i.e. if 1 is true the
        # rest can't be and 2 can only be true if 1 isn't.
        # Start with checking if last iteration was case_5, and we need
        # to add another point to the refined grid.
        case_1 = last_point_intersect
        case_2 = is_this_the_last_point * (1 - case_1)
        case_3 = suboptimal_cond * (1 - case_1) * (1 - case_2)
        case_4 = ~switch_value_func * (1 - case_1) * (1 - case_2) * (1 - case_3)
        case_5 = (
            next_point_past_intersect
            * (1 - case_1)
            * (1 - case_2)
            * (1 - case_3)
            * (1 - case_4)
        )
        case_6 = (
            point_j_past_intersect
            * (1 - case_1)
            * (1 - case_2)
            * (1 - case_3)
            * (1 - case_4)
            * (1 - case_5)
        )

        (
            intersect_grid,
            intersect_value,
            intersect_policy_left,
            intersect_policy_right,
        ) = select_and_calculate_intersection(
            endog_grid=endog_grid,
            policy=policy,
            value=value,
            endog_grid_k_and_j=endog_grid_k_and_j,
            value_k_and_j=value_k_and_j,
            policy_k_and_j=policy_k_and_j,
            idx_next_on_lower_curve=idx_next_on_lower_curve,
            idx_before_on_upper_curve=idx_before_on_upper_curve,
            idx_to_inspect=idx_to_inspect,
            case_5=case_5,
            case_6=case_6,
        )

        # Save the values for the next iteration
        (
            value_to_save,
            policy_left_to_save,
            policy_right_to_save,
            endog_grid_to_save,
        ) = select_variables_to_save_this_iteration(
            case_6=case_6,
            intersect_value=intersect_value,
            intersect_policy_left=intersect_policy_left,
            intersect_policy_right=intersect_policy_right,
            intersect_grid=intersect_grid,
            value_to_be_saved_next=value_to_be_saved_next,
            policy_left_to_be_saved_next=policy_left_to_be_saved_next,
            policy_right_to_be_saved_next=policy_right_to_be_saved_next,
            endog_grid_to_be_saved_next=endog_grid_to_be_saved_next,
        )
        value_case_2 = possible_values_case_2[idx_case_2]
        policy_to_be_saved_case_2 = possible_policies_case_2[idx_case_2]
        endog_grid_to_be_saved_case_2 = possible_endog_grid_case_2[idx_case_2]

        # In the iteration where case_2 is first time True, the last point is selected
        # and afterwards only nans.
        idx_case_2 = case_2
        last_point_intersect = case_5

        in_case_134 = case_1 + case_3 + case_4
        in_case_256 = case_2 + case_5 + case_6

        in_case_123 = case_1 + case_2 + case_3
        in_case_1236 = case_1 + case_2 + case_3 + case_6
        in_case_45 = case_4 + case_5

        in_case_146 = case_1 + case_4 + case_6

        value_to_be_saved_next = (
            in_case_146 * value[idx_to_inspect]
            + case_2 * value_case_2
            + case_5 * intersect_value
            + case_3 * value_to_be_saved_next
        )
        policy_left_to_be_saved_next = (
            in_case_146 * policy[idx_to_inspect]
            + case_2 * policy_to_be_saved_case_2
            + case_5 * intersect_policy_left
            + case_3 * policy_left_to_be_saved_next
        )
        policy_right_to_be_saved_next = (
            in_case_146 * policy[idx_to_inspect]
            + case_2 * policy_to_be_saved_case_2
            + case_5 * intersect_policy_right
            + case_3 * policy_right_to_be_saved_next
        )
        endog_grid_to_be_saved_next = (
            in_case_146 * endog_grid[idx_to_inspect]
            + case_2 * endog_grid_to_be_saved_case_2
            + case_5 * intersect_grid
            + case_3 * endog_grid_to_be_saved_next
        )

        # In case 1, 2, 3 the old value remains as value_j, in 4, 5, value_j is former
        # value k and in 6 the old value_j is overwritten
        value_j_new = (
            in_case_123 * value_k_and_j[1]
            + in_case_45 * value[idx_to_inspect]
            + case_6 * intersect_value
        )
        value_k_new = in_case_1236 * value_k_and_j[0] + in_case_45 * value_k_and_j[1]
        value_k_and_j = value_k_new, value_j_new
        policy_j_new = (
            in_case_123 * policy_k_and_j[1]
            + in_case_45 * policy[idx_to_inspect]
            + case_6 * intersect_policy_right
        )
        policy_k_new = in_case_1236 * policy_k_and_j[0] + in_case_45 * policy_k_and_j[1]
        policy_k_and_j = policy_k_new, policy_j_new
        endog_grid_j_new = (
            in_case_123 * endog_grid_k_and_j[1]
            + in_case_45 * endog_grid[idx_to_inspect]
            + case_6 * intersect_grid
        )
        endog_grid_k_new = (
            in_case_1236 * endog_grid_k_and_j[0] + in_case_45 * endog_grid_k_and_j[1]
        )
        endog_grid_k_and_j = endog_grid_k_new, endog_grid_j_new
        # Increase in cases 134 and not in 256
        idx_to_inspect += in_case_134 * (1 - in_case_256)

        value_refined[idx_refined] = value_to_save
        policy_left_refined[idx_refined] = policy_left_to_save
        policy_right_refined[idx_refined] = policy_right_to_save
        endog_grid_refined[idx_refined] = endog_grid_to_save

    return value_refined, policy_left_refined, policy_right_refined, endog_grid_refined


# def scan_body():


def select_variables_to_save_this_iteration(
    case_6,
    intersect_value,
    intersect_policy_left,
    intersect_policy_right,
    intersect_grid,
    value_to_be_saved_next,
    policy_left_to_be_saved_next,
    policy_right_to_be_saved_next,
    endog_grid_to_be_saved_next,
):
    # Determine variables to save this iteration. This is always the variables
    # carried from last iteration. Except in case 6.
    value_to_save = value_to_be_saved_next * (1 - case_6) + intersect_value * case_6
    policy_left_to_save = (
        policy_left_to_be_saved_next * (1 - case_6) + intersect_policy_left * case_6
    )
    policy_right_to_save = (
        policy_right_to_be_saved_next * (1 - case_6) + intersect_policy_right * case_6
    )
    endog_grid_to_save = (
        endog_grid_to_be_saved_next * (1 - case_6) + intersect_grid * case_6
    )
    return value_to_save, policy_left_to_save, policy_right_to_save, endog_grid_to_save


def select_and_calculate_intersection(
    endog_grid,
    policy,
    value,
    endog_grid_k_and_j,
    value_k_and_j,
    policy_k_and_j,
    idx_next_on_lower_curve,
    idx_before_on_upper_curve,
    idx_to_inspect,
    case_5,
    case_6,
):
    wealth_1_on_lower_curve = (
        endog_grid[idx_next_on_lower_curve] * case_5 + endog_grid_k_and_j[0] * case_6
    )
    value_1_on_lower_curve = (
        value[idx_next_on_lower_curve] * case_5 + value_k_and_j[0] * case_6
    )
    policy_1_on_lower_curve = (
        policy[idx_next_on_lower_curve] * case_5 + policy_k_and_j[0] * case_6
    )
    (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    ) = calc_intersection_and_extrapolate_policy(
        wealth_1_lower_curve=wealth_1_on_lower_curve,
        value_1_lower_curve=value_1_on_lower_curve,
        policy_1_lower_curve=policy_1_on_lower_curve,
        wealth_2_lower_curve=endog_grid_k_and_j[1],
        value_2_lower_curve=value_k_and_j[1],
        policy_2_lower_curve=policy_k_and_j[1],
        wealth_1_upper_curve=endog_grid[idx_to_inspect],
        value_1_upper_curve=value[idx_to_inspect],
        policy_1_upper_curve=policy[idx_to_inspect],
        wealth_2_upper_curve=endog_grid[idx_before_on_upper_curve],
        value_2_upper_curve=value[idx_before_on_upper_curve],
        policy_2_upper_curve=policy[idx_before_on_upper_curve],
    )
    return (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    )


def calc_intersection_and_extrapolate_policy(
    wealth_1_lower_curve,
    value_1_lower_curve,
    policy_1_lower_curve,
    wealth_2_lower_curve,
    value_2_lower_curve,
    policy_2_lower_curve,
    wealth_1_upper_curve,
    value_1_upper_curve,
    policy_1_upper_curve,
    wealth_2_upper_curve,
    value_2_upper_curve,
    policy_2_upper_curve,
):
    """Calculate intersection of two lines and extrapolate policy.

    Args:
        wealth_1_lower_curve (float):
        value_1_lower_curve (float):
        policy_1_lower_curve (float):
        wealth_2_lower_curve (float):
        value_2_lower_curve (float):
        policy_2_lower_curve (float):
        wealth_1_upper_curve (float):
        value_1_upper_curve (float):
        policy_1_upper_curve (float):
        wealth_2_upper_curve (float):
        value_2_upper_curve (float):
        policy_2_upper_curve (float):

    Returns:
        Tuple[float, float, float, float]: intersection point on wealth grid, value
            function at intersection and on lower as well as upper curve extrapolated
            policy function.

    """
    # Calculate intersection of two lines
    intersect_grid, intersect_value = _linear_intersection(
        x1=wealth_1_lower_curve,
        y1=value_1_lower_curve,
        x2=wealth_2_lower_curve,
        y2=value_2_lower_curve,
        x3=wealth_1_upper_curve,
        y3=value_1_upper_curve,
        x4=wealth_2_upper_curve,
        y4=value_2_upper_curve,
    )

    # Extrapolate policy
    policy_left = _evaluate_point_on_line(
        x1=wealth_1_lower_curve,
        y1=policy_1_lower_curve,
        x2=wealth_2_lower_curve,
        y2=policy_2_lower_curve,
        point_to_evaluate=intersect_grid,
    )

    policy_right = _evaluate_point_on_line(
        x1=wealth_1_upper_curve,
        y1=policy_1_upper_curve,
        x2=wealth_2_upper_curve,
        y2=policy_2_upper_curve,
        point_to_evaluate=intersect_grid,
    )

    return intersect_grid, intersect_value, policy_left, policy_right


# @njit
def _forward_scan(
    value: np.ndarray,
    endog_grid: np.ndarray,
    policy: np.array,
    jump_thresh: float,
    endog_grid_current: float,
    exog_grid_current: float,
    idx_base: int,
    n_points_to_scan: int,
) -> Tuple[float, int]:
    """Scan forward to check which point is on same value function as idx_base.

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

    Returns:
        tuple:

        - grad_next_forward (float): The gradient of the next point on the same
            value function.
        - idx_on_same_value (int): Index of next point on the value function.

    """

    found_next_value_already = 0
    idx_on_same_value = 0
    grad_next_on_same_value = 0

    idx_max = endog_grid.shape[0] - 1

    for i in range(1, n_points_to_scan + 1):
        # Avoid out of bound indexing
        idx_to_check = min(idx_base + i, idx_max)
        # Get endog grid diff from current optimal to the one checkec
        endog_grid_diff = endog_grid_current - endog_grid[idx_to_check] + 1e-16
        # Check if checked point is on the same value function
        exog_grid_idx_to_check = endog_grid[idx_to_check] - policy[idx_to_check]
        is_on_same_value = (
            np.abs((exog_grid_current - exog_grid_idx_to_check) / (endog_grid_diff))
            < jump_thresh
        )
        gradient_next_denominator = (
            endog_grid[idx_base] - endog_grid[idx_to_check] + -1e-16
        )
        # Calculate gradient
        gradient_next = (value[idx_base] - value[idx_to_check]) / (
            gradient_next_denominator
        )

        # Now check if this is the first value on the same value function
        # This is only 1 if so far there hasn't been found a point and the point is on
        # the same value function
        value_is_next_on_same_value = is_on_same_value * (1 - found_next_value_already)
        # Update if you have found a point. Always 1 (=True) if you have found a point
        # already
        found_next_value_already = logic_or(found_next_value_already, is_on_same_value)

        # Update the index the first time a point is found
        idx_on_same_value += idx_to_check * value_is_next_on_same_value

        # Update the gradient the first time a point is found
        grad_next_on_same_value += gradient_next * value_is_next_on_same_value

    return (
        grad_next_on_same_value,
        idx_on_same_value,
    )


def logic_or(bool_ind_1, bool_ind_2):
    """Logical or function.

    Args:
        bool_ind_1 (np.ndarray): 1d array of booleans.
        bool_ind_2 (np.ndarray): 1d array of booleans.

    Returns:
        np.ndarray: 1d array of booleans.

    """
    both = bool_ind_1 * bool_ind_2
    either = bool_ind_1 + bool_ind_2
    return both + (1 - both) * either


# @njit
def _backward_scan(
    value: np.ndarray,
    endog_grid: np.ndarray,
    policy: np.array,
    jump_thresh: float,
    endog_grid_current,
    value_current,
    idx_base: int,
    n_points_to_scan: int,
) -> Tuple[float, int]:
    """Find point on same value function to idx_base.

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
        idx_base (int): Index of the base point in the value function to which find a
            point before on the same value function.

    Returns:
        tuple:

        - grad_before_on_same_value (float): The gradient of the previous point on
            the same value function.
        - is_before_on_same_value (int): Indicator for whether we have found a
            previous point on the same value function.

    """

    found_value_before_already = 0
    idx_point_before_on_same_value = 0
    grad_before_on_same_value = 0

    for i in range(1, n_points_to_scan + 1):
        idx_to_check = max(idx_base - i, 0)

        endog_grid_diff_to_current = np.maximum(
            endog_grid_current - endog_grid[idx_to_check], 1e-16
        )
        endog_grid_diff_to_next = np.maximum(
            endog_grid[idx_base] - endog_grid[idx_to_check], 1e-16
        )
        exog_grid_idx_base = endog_grid[idx_base] - policy[idx_base]
        exog_grid_idx_to_check = endog_grid[idx_to_check] - policy[idx_to_check]
        is_on_same_value = (
            np.abs(
                (exog_grid_idx_base - exog_grid_idx_to_check) / endog_grid_diff_to_next
            )
            < jump_thresh
        )
        grad_before = (value_current - value[idx_to_check]) / endog_grid_diff_to_current
        # Now check if this is the first value on the same value function
        # This is only 1 if so far there hasn't been found a point and the point is on
        # the same value function
        is_before = is_on_same_value * (1 - found_value_before_already)
        # Update if you have found a point. Always 1 (=True) if you have found a point
        # already
        found_value_before_already = logic_or(
            found_value_before_already, is_on_same_value
        )

        # Update the first time a new point is found
        idx_point_before_on_same_value += idx_to_check * is_before

        # Update the first time a new point is found
        grad_before_on_same_value += grad_before * is_before

    return (
        grad_before_on_same_value,
        idx_point_before_on_same_value,
    )


@njit
def _evaluate_point_on_line(
    x1: float, y1: float, x2: float, y2: float, point_to_evaluate: float
) -> float:
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
    return (y2 - y1) / ((x2 - x1) + 1e-16) * (point_to_evaluate - x1) + y1


@njit
def _linear_intersection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> Tuple[float, float]:
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

    slope1 = (y2 - y1) / ((x2 - x1) + 1e-16)
    slope2 = (y4 - y3) / ((x4 - x3) + 1e-16)

    x_intersection = (slope1 * x1 - slope2 * x3 + y3 - y1) / ((slope1 - slope2) + 1e-16)
    y_intersection = slope1 * (x_intersection - x1) + y1

    return x_intersection, y_intersection


def _augment_grids(
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    choice: int,
    expected_value_zero_savings: np.ndarray,
    min_wealth_grid: float,
    points_to_add: int,
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
        expected_value_zero_savings (float): The agent's expected value given that she
            saves zero.
        min_wealth_grid (float): Minimal wealth level in the endogenous wealth grid.
        points_to_add (int): Number of grid points to add. Roughly num_wealth / 10.
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
    grid_points_to_add = np.linspace(min_wealth_grid, endog_grid[0], points_to_add)[:-1]

    grid_augmented = np.append(grid_points_to_add, endog_grid)
    values_to_add = compute_value(
        grid_points_to_add,
        expected_value_zero_savings,
        choice,
    )
    value_augmented = np.append(values_to_add, value)
    policy_augmented = np.append(grid_points_to_add, policy)

    return grid_augmented, value_augmented, policy_augmented


def _initialize_refined_arrays(
    value: np.ndarray, policy: np.ndarray, endog_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value_refined = np.empty(shape=int(1.2 * len(value)))
    policy_left_refined = np.empty_like(value_refined)
    policy_right_refined = np.empty_like(value_refined)
    endog_grid_refined = np.empty_like(value_refined)

    return value_refined, policy_left_refined, policy_right_refined, endog_grid_refined
