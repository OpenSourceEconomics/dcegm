"""Extension of the Fast Upper-Envelope Scan.

The original algorithm is based on Loretti I. Dobrescu and Akshay Shanker (2022) 'Fast
Upper-Envelope Scan for Solving Dynamic Optimization Problems',
https://dx.doi.org/10.2139/ssrn.4181302

"""
from functools import partial
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp  # noqa: F401
from jax import jit  # noqa: F401
from jax import vmap


def fast_upper_envelope_wrapper(
    endog_grid: jnp.ndarray,
    policy: jnp.ndarray,
    value: jnp.ndarray,
    expected_value_zero_savings: float,
    choice: int,
    params: Dict[str, float],
    compute_value: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        expected_value_zero_savings (float): The agent's expected value given that she
            saves zero.
        choice (int): The current choice.
        compute_value (callable): Function to compute the agent's value.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - endog_grid_refined (np.ndarray): 1d array of shape (1.1 * n_grid_wealth,)
            containing the refined state- and choice-specific endogenous grid.
        - policy_refined_with_nans (np.ndarray): 1d array of shape (1.1 * n_grid_wealth)
            containing refined state- and choice-specificconsumption policy.
        - value_refined_with_nans (np.ndarray): 1d array of shape (1.1 * n_grid_wealth)
            containing refined state- and choice-specific value function.

    """
    min_wealth_grid = jnp.min(endog_grid)
    # These tuning parameters should be set outside. Don't want to touch solve.py now
    points_to_add = len(endog_grid) // 10
    num_iter = int(1.2 * value.shape[0])
    jump_thresh = 2
    # Non-concave region coincides with credit constraint.
    # This happens when there is a non-monotonicity in the endogenous wealth grid
    # that goes below the first point.
    # Solution: Value function to the left of the first point is analytical,
    # so we just need to add some points to the left of the first grid point.
    # We do that independent of whether the condition is fulfilled or not.
    # If the condition is not fulfilled this is points_to_add times the same point.

    # This is the condition, which we do not use at the moment.
    # closed_form_cond = min_wealth_grid < endog_grid[0]
    grid_points_to_add = jnp.linspace(min_wealth_grid, endog_grid[0], points_to_add)[
        :-1
    ]
    values_to_add = vmap(compute_value, in_axes=(0, None, None, None))(
        grid_points_to_add,
        expected_value_zero_savings,
        choice,
        params,
    )

    grid_augmented = jnp.append(grid_points_to_add, endog_grid)
    value_augmented = jnp.append(values_to_add, value)
    policy_augmented = jnp.append(grid_points_to_add, policy)

    (
        endog_grid_refined,
        value_refined,
        policy_left_refined,
        policy_right_refined,
    ) = fast_upper_envelope(
        grid_augmented,
        value_augmented,
        policy_augmented,
        expected_value_zero_savings,
        num_iter=num_iter,
        jump_thresh=jump_thresh,
    )
    return (
        endog_grid_refined,
        policy_left_refined,
        policy_right_refined,
        value_refined,
    )


def fast_upper_envelope(
    endog_grid: jnp.ndarray,
    value: jnp.ndarray,
    policy: jnp.ndarray,
    expected_value_zero_savings: float,
    num_iter: int,
    jump_thresh: Optional[float] = 2,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Remove suboptimal points from the endogenous grid, policy, and value function.

    Args:
        endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
            grid of shape (n_grid_wealth + 1,).
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth + 1,).
        policy (np.ndarray): 1d array containing the unrefined policy correspondence
            of shape (n_grid_wealth + 1,).
        expected_value_zero_savings (float): The agent's expected value given that she
            saves zero.
        num_iter (int): Number of iterations to execute the fues. Recommended to use
            twenty percent more than the actual array size.
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
    # Comment by Akshay: Determine locations where endogenous grid points are
    # equal to the lower bound. Not relevant for us.
    # mask = endog_grid <= lower_bound_wealth
    # if jnp.any(mask):
    #     max_value_lower_bound = jnp.nanmax(value[mask])
    #     mask &= value < max_value_lower_bound
    #     value[mask] = jnp.nan

    idx_sort = jnp.argsort(endog_grid)
    value = jnp.take(value, idx_sort)
    policy = jnp.take(policy, idx_sort)
    endog_grid = jnp.take(endog_grid, idx_sort)

    (
        value_refined,
        policy_left_refined,
        policy_right_refined,
        endog_grid_refined,
    ) = scan_value_function(
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        expected_value_zero_savings=expected_value_zero_savings,
        num_iter=num_iter,
        jump_thresh=jump_thresh,
        n_points_to_scan=10,
    )

    return endog_grid_refined, value_refined, policy_left_refined, policy_right_refined


def scan_value_function(
    endog_grid: jnp.ndarray,
    value: jnp.ndarray,
    policy: jnp.ndarray,
    expected_value_zero_savings,
    num_iter: int,
    jump_thresh: float,
    n_points_to_scan: Optional[int] = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan the value function to remove suboptimal points and add intersection points.

    Args:
        endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
            grid of shape (n_grid_wealth + 1,).
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth + 1,).
        policy (np.ndarray): 1d array containing the unrefined policy correspondence
            of shape (n_grid_wealth + 1,).
        expected_value_zero_savings (float): The agent's expected value given that she
            saves zero.
        num_iter (int): Number of iterations to execute the fues. Recommended to use
            twenty percent more than the actual array size.
        jump_thresh (float): Jump detection threshold.
        n_points_to_scan (int): Number of points to scan for suboptimal points.

    Returns:
        tuple:

        - (np.ndarray): 1d array of shape (n_grid_clean,) containing the refined
            value function. Overlapping segments have been removed and only
            the optimal points are kept.

    """
    value_k_and_j = expected_value_zero_savings, value[0]
    endog_grid_k_and_j = 0, endog_grid[0]
    policy_k_and_j = 0, policy[0]
    vars_j_and_k_inital = (value_k_and_j, policy_k_and_j, endog_grid_k_and_j)

    to_be_saved_inital = (expected_value_zero_savings, 0.0, 0.0, 0.0)

    idx_to_inspect = 0
    last_point_was_intersect = False
    saved_last_point_already = False

    carry_init = (
        vars_j_and_k_inital,
        to_be_saved_inital,
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
    )
    partial_body = partial(
        scan_body,
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        jump_thresh=jump_thresh,
        n_points_to_scan=n_points_to_scan,
    )

    _final_carry, result = jax.lax.scan(
        partial_body,
        carry_init,
        xs=None,
        length=num_iter,
    )

    return result


def scan_body(
    carry,
    _iter_step,
    value,
    policy,
    endog_grid,
    jump_thresh,
    n_points_to_scan,
):
    """This is the body exucted at each iteration of the scan function. Depending on the
    idx_to_inspect of the carry value it scans either a new value or just saves the
    value from last period. The carry value is updated in each iteration and passed to
    the next iteration. This body returns one value, two policy values (left and right)
    as well as an endogenous grid value.

        Args:
            carry (tuple): The carry value passed from the previous iteration. This is a
                tuple containing the variables that are updated in each iteration.
                Including the current two optimal points j and k, the points to be saved
                this iteration as well as the indexes of the point to be inspected, the
                indicator of case 2 and the indicator if the last point was an
                interseqction point.
            _iter_step (int): The count of iteration we are in.
            value (np.ndarray): 1d array containing the unrefined value correspondence
                of shape (n_grid_wealth,).
            policy (np.ndarray): 1d array containing the unrefined policy correspondence
                of shape (n_grid_wealth,).
            endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
                grid of shape (n_grid_wealth,).
            jump_thresh (float): Jump detection threshold.
            n_points_to_scan (int): Number of points to scan in forward and backwards
                scan.

    Returns:
        tuple:

        - carry (tuple): The updated carry value passed to the next iteration.
        - result (tuple): The result of this iteration. This is a tuple containing the
            value, the left and right policy as well as the endogenous grid value to be
            saved this iteration.

    """
    (
        vars_j_and_k,
        planed_to_be_saved_this_iter,
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
    ) = carry
    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = vars_j_and_k

    is_this_the_last_point = idx_to_inspect == len(endog_grid) - 1
    # In each iteration we calculate the gradient of the value function

    (
        grad_next_forward,
        idx_next_on_lower_curve,
    ) = _forward_scan(
        value=value,
        endog_grid=endog_grid,
        policy=policy,
        jump_thresh=jump_thresh,
        endog_grid_current=endog_grid_k_and_j[1],
        policy_current=policy_k_and_j[1],
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

    cases, suboptimal_cond = create_cases(
        value_idx_to_inspect=value[idx_to_inspect],
        policy_idx_to_inspect=policy[idx_to_inspect],
        endog_grid_idx_to_inspect=endog_grid[idx_to_inspect],
        value_k_and_j=value_k_and_j,
        policy_k_and_j=policy_k_and_j,
        endog_grid_k_and_j=endog_grid_k_and_j,
        grad_next_forward=grad_next_forward,
        grad_next_backward=grad_next_backward,
        last_point_was_intersect=last_point_was_intersect,
        is_this_the_last_point=is_this_the_last_point,
        jump_thresh=jump_thresh,
    )

    intersection_point = select_and_calculate_intersection(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        endog_grid_k_and_j=endog_grid_k_and_j,
        value_k_and_j=value_k_and_j,
        policy_k_and_j=policy_k_and_j,
        idx_next_on_lower_curve=idx_next_on_lower_curve,
        idx_before_on_upper_curve=idx_before_on_upper_curve,
        idx_to_inspect=idx_to_inspect,
        case_5=cases[4],
        case_6=cases[5],
    )

    # Save the values for the next iteration
    result_to_save_this_iteration = select_variables_to_save_this_iteration(
        case_6=cases[5],
        intersection_point=intersection_point,
        planed_to_be_saved_this_iter=planed_to_be_saved_this_iter,
    )

    variables_to_be_saved_next_iteration = select_points_to_be_saved_next_iteration(
        value_idx_to_inspect=value[idx_to_inspect],
        policy_idx_to_inspect=policy[idx_to_inspect],
        endog_grid_idx_to_inspect=endog_grid[idx_to_inspect],
        value_case_2=jax.lax.select(saved_last_point_already, jnp.nan, value[-1]),
        policy_case_2=jax.lax.select(saved_last_point_already, jnp.nan, policy[-1]),
        endog_grid_case_2=jax.lax.select(
            saved_last_point_already, jnp.nan, endog_grid[-1]
        ),
        intersection_point=intersection_point,
        planed_to_be_saved_this_iter=planed_to_be_saved_this_iter,
        cases=cases,
    )
    case_1, case_2, case_3, case_4, case_5, case_6 = cases
    (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    ) = intersection_point

    # In the iteration where case_2 is first time True, the last point is selected
    # and afterwards only nans.
    saved_last_point_already = case_2
    last_point_was_intersect = case_5

    in_case_134 = case_1 | case_3 | case_4
    in_case_256 = ~in_case_134
    in_case_123 = case_1 | case_2 | case_3
    in_case_1236 = case_1 | case_2 | case_3 | case_6
    in_case_45 = case_4 | case_5

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
    update_by_cases = in_case_134 * (1 - in_case_256)
    idx_to_inspect += update_by_cases + suboptimal_cond * (1 - update_by_cases)
    vars_j_and_k = (value_k_and_j, policy_k_and_j, endog_grid_k_and_j)
    carry = (
        vars_j_and_k,
        variables_to_be_saved_next_iteration,
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
    )

    return carry, result_to_save_this_iteration


def select_points_to_be_saved_next_iteration(
    value_idx_to_inspect,
    policy_idx_to_inspect,
    endog_grid_idx_to_inspect,
    value_case_2,
    policy_case_2,
    endog_grid_case_2,
    intersection_point,
    planed_to_be_saved_this_iter,
    cases,
):
    case_1, case_2, case_3, case_4, case_5, case_6 = cases
    (
        planed_value,
        planed_policy_left,
        planed_policy_right,
        planed_endog_grid,
    ) = planed_to_be_saved_this_iter

    (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    ) = intersection_point
    in_case_146 = case_1 | case_4 | case_6

    value_to_be_saved_next = (
        in_case_146 * value_idx_to_inspect
        + case_2 * value_case_2
        + case_5 * intersect_value
        + case_3 * planed_value
    )
    policy_left_to_be_saved_next = (
        in_case_146 * policy_idx_to_inspect
        + case_2 * policy_case_2
        + case_5 * intersect_policy_left
        + case_3 * planed_policy_left
    )
    policy_right_to_be_saved_next = (
        in_case_146 * policy_idx_to_inspect
        + case_2 * policy_case_2
        + case_5 * intersect_policy_right
        + case_3 * planed_policy_right
    )
    endog_grid_to_be_saved_next = (
        in_case_146 * endog_grid_idx_to_inspect
        + case_2 * endog_grid_case_2
        + case_5 * intersect_grid
        + case_3 * planed_endog_grid
    )
    return (
        value_to_be_saved_next,
        policy_left_to_be_saved_next,
        policy_right_to_be_saved_next,
        endog_grid_to_be_saved_next,
    )


def create_cases(
    value_idx_to_inspect,
    policy_idx_to_inspect,
    endog_grid_idx_to_inspect,
    value_k_and_j,
    policy_k_and_j,
    endog_grid_k_and_j,
    grad_next_forward,
    grad_next_backward,
    last_point_was_intersect,
    is_this_the_last_point,
    jump_thresh,
):
    grad_before = calculate_gradient(
        x1=endog_grid_k_and_j[1],
        y1=value_k_and_j[1],
        x2=endog_grid_k_and_j[0],
        y2=value_k_and_j[0],
    )

    # gradient with leading index to be checked
    grad_next = calculate_gradient(
        x1=endog_grid_idx_to_inspect,
        y1=value_idx_to_inspect,
        x2=endog_grid_k_and_j[1],
        y2=value_k_and_j[1],
    )

    suboptimal_cond, does_the_value_func_switch = check_for_suboptimality(
        endog_grid_k_and_j=endog_grid_k_and_j,
        policy_k_and_j=policy_k_and_j,
        value_k_and_j=value_k_and_j,
        value_idx_to_inspect=value_idx_to_inspect,
        endog_grid_idx_to_inspect=endog_grid_idx_to_inspect,
        policy_idx_to_inspect=policy_idx_to_inspect,
        grad_next=grad_next,
        grad_next_forward=grad_next_forward,
        grad_before=grad_before,
        jump_thresh=jump_thresh,
    )

    next_point_past_intersect = (grad_before > grad_next) | (
        grad_next < grad_next_backward
    )
    point_j_past_intersect = grad_next > grad_next_backward

    # Generate cases. They are exclusive in ascending order, i.e. if 1 is true the
    # rest can't be and 2 can only be true if 1 isn't.
    # Start with checking if last iteration was case_5, and we need
    # to add another point to the refined grid.
    case_1 = last_point_was_intersect
    case_2 = is_this_the_last_point & ~case_1
    case_3 = suboptimal_cond & ~case_1 & ~case_2
    case_4 = ~does_the_value_func_switch * ~case_1 * ~case_2 * ~case_3
    case_5 = next_point_past_intersect & ~case_1 & ~case_2 & ~case_3 & ~case_4
    case_6 = point_j_past_intersect & ~case_1 & ~case_2 & ~case_3 & ~case_4 & ~case_5
    return (case_1, case_2, case_3, case_4, case_5, case_6), suboptimal_cond


def check_for_suboptimality(
    endog_grid_k_and_j,
    policy_k_and_j,
    value_k_and_j,
    value_idx_to_inspect,
    endog_grid_idx_to_inspect,
    policy_idx_to_inspect,
    grad_next,
    grad_next_forward,
    grad_before,
    jump_thresh,
):
    """This function checks for sub-optimality of the current point.

    If this function   returns False the point can still be suboptimal. That is if we
    find in the   next iteration, that this point actually is after a switch point. Here
    we check   if the point fulfills one of three conditions. Either the value function
    is   decreasing with decreasing value function, the # value function not montone in
    consumption or # if the gradient of the index we inspect and the point j (the last
    point # on the same choice specific policy) is shallower than the # gradient joining
    the i+1 and j, then delete j'th point # If the point is the same as point j, this is
    always false and # switch_value_func as well. Therefore, the third if is chosen.

    """
    does_the_value_func_switch = create_indicator_if_value_function_is_switched(
        endog_grid_1=endog_grid_k_and_j[1],
        policy_1=policy_k_and_j[1],
        endog_grid_2=endog_grid_idx_to_inspect,
        policy_2=policy_idx_to_inspect,
        jump_thresh=jump_thresh,
    )
    switch_value_func_and_steep_increase_after = (
        grad_next < grad_next_forward
    ) & does_the_value_func_switch

    decreasing_value = value_idx_to_inspect < value_k_and_j[1]

    are_savings_non_monotone = check_for_non_monotone_savings(
        endog_grid_j=endog_grid_k_and_j[1],
        policy_j=policy_k_and_j[1],
        endog_grid_idx_to_inspect=endog_grid_idx_to_inspect,
        policy_idx_to_inspect=policy_idx_to_inspect,
    )

    # Aggregate the three cases
    suboptimal_cond = (
        switch_value_func_and_steep_increase_after
        | decreasing_value
        # Do we need the grad condition next?
        | (are_savings_non_monotone & (grad_next < grad_before))
    )
    return suboptimal_cond, does_the_value_func_switch


def check_for_non_monotone_savings(
    endog_grid_j, policy_j, endog_grid_idx_to_inspect, policy_idx_to_inspect
):
    """This function checks if the savings are a non-monotone in wealth between the
    current last point on the upper envelope j and the point we check.

    Args:
        endog_grid_j (float): The endogenous grid value of the last point on the upper
            envelope.
        policy_j (float): The policy value of the last point on the upper envelope.
        endog_grid_idx_to_inspect (float): The endogenous grid value of the point we
            check.
        policy_idx_to_inspect (float): The policy value of the point we check.

    Returns:
        non_monotone_policy (bool): Indicator if the policy is non-monotone in wealth
            between the last point on the upper envelope and the point we check.

    """
    exog_grid_j = endog_grid_j - policy_j
    exog_grid_idx_to_inspect = endog_grid_idx_to_inspect - policy_idx_to_inspect
    are_savings_non_monotone = exog_grid_idx_to_inspect < exog_grid_j
    return are_savings_non_monotone


def select_variables_to_save_this_iteration(
    case_6,
    intersection_point,
    planed_to_be_saved_this_iter,
):
    """This function selects depending on the case we are in, the value which is saved
    this iteration.

    This is always the point which was set last period to be saved this period except in
    case 6, where we realize that this point actually needs to be disregarded.

    """
    (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    ) = intersection_point
    (
        planed_value,
        planed_policy_left,
        planed_policy_right,
        planed_endog_grid,
    ) = planed_to_be_saved_this_iter
    # Determine variables to save this iteration. This is always the variables
    # carried from last iteration. Except in case 6.
    value_to_save = planed_value * (1 - case_6) + intersect_value * case_6
    policy_left_to_save = (
        planed_policy_left * (1 - case_6) + intersect_policy_left * case_6
    )
    policy_right_to_save = (
        planed_policy_right * (1 - case_6) + intersect_policy_right * case_6
    )
    endog_grid_to_save = planed_endog_grid * (1 - case_6) + intersect_grid * case_6
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
    """This function selects the points with which we can compute to the intersection
    points.

    This functions maps very nicely into Figure 5 of the paper. In case 5, we use the
    next point (q in the graph) we found on the value function segment of point j(i in
    graph) and intersect it with the idx_to_check (i + 1 in the graph). In case 6 we
    are in the situation of the figure in the right site in Figure 5. Here we intersect
    the line of j and k (i and i-1 in the graph) with the line of idx_to_check (i+1
    in the graph) and the point before on the same value segment.

    Args:
        endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
            grid of shape (n_grid_wealth,).
        policy (np.ndarray): 1d array containing the unrefined policy correspondence
            of shape (n_grid_wealth,).
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth,).
        endog_grid_k_and_j (tuple): Tuple containing the endogenous grid values of the
            last two points on the upper envelope.
        value_k_and_j (tuple): Tuple containing the value function values of the last
            two points on the upper envelope.
        policy_k_and_j (tuple): Tuple containing the policy function values of the last
            two points on the upper envelope.
        idx_next_on_lower_curve (int): The index of the next point on the lower curve.
        idx_before_on_upper_curve (int): The index of the point before on the upper
            curve.
        idx_to_inspect (int): The index of the point to inspect.
        case_5 (bool): Indicator if we are in case 5.
        case_6 (bool): Indicator if we are in case 6.

    Returns:
        intersect_grid (float): The endogenous grid value of the intersection point.
        intersect_value (float): The value function value of the intersection point.
        intersect_policy_left (float): The policy function value of the left continuous
            of the policy function at the intersection point.
        intersect_policy_right (float): The policy function value of the right
            continuous of the policy function at the intersection point.

    """
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
    wealth_1_lower_curve: float,
    value_1_lower_curve: float,
    policy_1_lower_curve: float,
    wealth_2_lower_curve: float,
    value_2_lower_curve: float,
    policy_2_lower_curve: float,
    wealth_1_upper_curve: float,
    value_1_upper_curve: float,
    policy_1_upper_curve: float,
    wealth_2_upper_curve: float,
    value_2_upper_curve: float,
    policy_2_upper_curve: float,
):
    """Calculate the intersection of the value functions and return the intersection
    wealth grid, the value function as well as the left and right policy values. Even
    though the function is described for scalars it also works for arrays.

    We introduce here the left and right policy function to avoid inserting two wealth
    points next to each other in comparison to the original upper_envelope. The left and
    right policy arrays coincide on all entries except the entry corresponding to the
    intersection point in the wealth grid. We use left and right as lower and higher
    on the line of all real numbers. The left policy function holds the extrapolated
    policy function value from the two points left to the intersection. We use this
    value if we interpolate the policy function for a wealth point between the wealth
    point to the left of the intersection point and the intersection point. The right
    policy function value vice versa holds the extrapolated policy function value from
    the two points right to the intersection point, and we use it to interpolate the
    policy if a wealth point is to the right of the intersection.

    Our interpolation function incorporates this structure and benefits from the fact
    that value and policy function uses the same wealth grid. More information is
    provided in the interpolation module.

    Args:
        wealth_1_lower_curve (float): The first wealth point on the lower curve.
        value_1_lower_curve (float): The value function at the first wealth point on the
            lower curve.
        policy_1_lower_curve (float): The policy function at the first wealth point on
            the lower curve.
        wealth_2_lower_curve (float): The second wealth point on the lower curve.
        value_2_lower_curve (float): The value function at the second wealth point on
            the lower curve.
        policy_2_lower_curve (float): The policy function at the second wealth point on
            the lower curve.
        wealth_1_upper_curve (float): The first wealth point on the upper curve.
        value_1_upper_curve (float): The value function at the first wealth point on
            the upper curve.
        policy_1_upper_curve (float): The policy function at the first wealth point on
            the upper curve.
        wealth_2_upper_curve (float): The second wealth point on the upper curve.
        value_2_upper_curve (float): The value function at the second wealth point on
            the     upper curve.
        policy_2_upper_curve (float): The policy function at the second wealth point on
            the upper curve.

    Returns:
        Tuple[float, float, float, float]: intersection point on wealth grid, value
            function at intersection and on lower as well as upper curve extrapolated
            policy function (left and right policy function).

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


def _forward_scan(
    value: jnp.ndarray,
    endog_grid: jnp.ndarray,
    policy: jnp.array,
    jump_thresh: float,
    endog_grid_current: float,
    policy_current: float,
    idx_base: int,
    n_points_to_scan: int,
) -> Tuple[float, int]:
    """Scan forward to check which point is on same value function as the current last
    point on the upper envelope.

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

    found_next_value_already = False
    idx_on_same_value = 0
    grad_next_on_same_value = 0

    idx_max = endog_grid.shape[0] - 1

    for i in range(1, n_points_to_scan + 1):
        # Avoid out of bound indexing
        idx_to_check = jnp.minimum(idx_base + i, idx_max)

        is_not_on_same_value = create_indicator_if_value_function_is_switched(
            endog_grid_1=endog_grid_current,
            policy_1=policy_current,
            endog_grid_2=endog_grid[idx_to_check],
            policy_2=policy[idx_to_check],
            jump_thresh=jump_thresh,
        )
        is_on_same_value = 1 - is_not_on_same_value
        gradient_next = calculate_gradient(
            x1=endog_grid[idx_base],
            y1=value[idx_base],
            x2=endog_grid[idx_to_check],
            y2=value[idx_to_check],
        )

        # Now check if this is the first value on the same value function
        # This is only 1 if so far there hasn't been found a point and the point is on
        # the same value function
        value_is_next_on_same_value = is_on_same_value & ~found_next_value_already
        # Update if you have found a point. Always 1 (=True) if you have found a point
        # already
        found_next_value_already = found_next_value_already | is_on_same_value

        # Update the index the first time a point is found
        idx_on_same_value += idx_to_check * value_is_next_on_same_value

        # Update the gradient the first time a point is found
        grad_next_on_same_value += gradient_next * value_is_next_on_same_value

    return (
        grad_next_on_same_value,
        idx_on_same_value,
    )


def calculate_gradient(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    """Calculate the gradient between two points. This function returns 0 if the points
    are the same.

    Args:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.

    Returns:
        float: The gradient between the two points.

    """
    denominator = x1 - x2 + 1e-16
    return (y1 - y2) / denominator


def logic_or(bool_ind_1, bool_ind_2):
    """Logical or function.

    Args:
        bool_ind_1 (bool): A bool
        bool_ind_2 (bool): A bool.

    Returns:
        bool: The logical or of the two bools.

    """
    both = bool_ind_1 * bool_ind_2
    either = bool_ind_1 + bool_ind_2
    return both + (1 - both) * either


def _backward_scan(
    value: jnp.ndarray,
    endog_grid: jnp.ndarray,
    policy: jnp.array,
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
        idx_to_check = jnp.maximum(idx_base - i, 0)

        is_not_on_same_value = create_indicator_if_value_function_is_switched(
            endog_grid_1=endog_grid[idx_base],
            policy_1=policy[idx_base],
            endog_grid_2=endog_grid[idx_to_check],
            policy_2=policy[idx_to_check],
            jump_thresh=jump_thresh,
        )
        is_on_same_value = 1 - is_not_on_same_value

        grad_before = calculate_gradient(
            x1=endog_grid_current,
            y1=value_current,
            x2=endog_grid[idx_to_check],
            y2=value[idx_to_check],
        )
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


def create_indicator_if_value_function_is_switched(
    endog_grid_1: float,
    policy_1: float,
    endog_grid_2: float,
    policy_2: float,
    jump_thresh: float,
):
    """Create an indicator if the value function is switched between two points.

    Args:
        endog_grid_1 (float): The first endogenous wealth point.
        policy_1 (float): The policy function at the first endogenous wealth point.
        endog_grid_2 (float): The second endogenous wealth point.
        policy_2 (float): The policy function at the second endogenous wealth point.

    Returns:
        bool: Indicator if value function is switched.

    """

    exog_grid_1 = endog_grid_1 - policy_1
    exog_grid_2 = endog_grid_2 - policy_2
    gradient_exog_grid = calculate_gradient(
        x1=endog_grid_1,
        y1=exog_grid_1,
        x2=endog_grid_2,
        y2=exog_grid_2,
    )
    gradient_exog_abs = jnp.abs(gradient_exog_grid)
    is_switched = gradient_exog_abs > jump_thresh
    return is_switched


def _evaluate_point_on_line(
    x1: float, y1: float, x2: float, y2: float, point_to_evaluate: float
) -> float:
    """Evaluate a point on a line defined by (x1, y1) and (x2, y2).

    Args:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.
        point_to_evaluate (float): The point to evaluate.

    Returns:
        float: The value of the point on the line.

    """
    return (y2 - y1) / ((x2 - x1) + 1e-16) * (point_to_evaluate - x1) + y1


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
    """Find the intersection of two lines defined by (x).

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

    slope1 = calculate_gradient(x1, y1, x2, y2)
    slope2 = calculate_gradient(x3, y3, x4, y4)

    x_intersection = (slope1 * x1 - slope2 * x3 + y3 - y1) / ((slope1 - slope2) + 1e-16)
    y_intersection = slope1 * (x_intersection - x1) + y1

    return x_intersection, y_intersection
