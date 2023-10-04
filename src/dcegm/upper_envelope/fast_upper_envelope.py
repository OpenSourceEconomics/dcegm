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
import jax.numpy as jnp
import numpy as np
from dcegm.upper_envelope.math_funcs import calc_gradient
from dcegm.upper_envelope.math_funcs import calc_intersection_and_extrapolate_policy
from jax import vmap


def fast_upper_envelope_wrapper(
    endog_grid: jnp.ndarray,
    policy: jnp.ndarray,
    value: jnp.ndarray,
    expected_value_zero_savings: float,
    state_choice_vec: jnp.ndarray,
    params: Dict[str, float],
    compute_utility: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Drop suboptimal points and refines the endogenous grid, policy, and value.

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
    min_wealth_grid = np.min(endog_grid)
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
    values_to_add = vmap(_compute_value, in_axes=(0, None, None, None, None))(
        grid_points_to_add,
        expected_value_zero_savings,
        state_choice_vec,
        params,
        compute_utility,
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


def _compute_value(
    consumption, next_period_value, state_choice_vec, params, compute_utility
):
    utility = compute_utility(
        consumption=consumption,
        params=params,
        **state_choice_vec,
    )
    return utility + params["beta"] * next_period_value


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
    last_point_in_grid = jnp.array([value[-1], policy[-1], endog_grid[-1]])
    dummy_points_grid = jnp.array([jnp.nan, jnp.nan, jnp.nan])

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
        last_point_in_grid=last_point_in_grid,
        dummy_points_grid=dummy_points_grid,
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
    last_point_in_grid,
    dummy_points_grid,
    jump_thresh,
    n_points_to_scan,
):
    """The scan body to be executed at each iteration of the scan function.

    Depending on the idx_to_inspect of the carry value, either a new value is scanned
    or the value from the last period is saved.
    The carry value is updated in each iteration and passed to the next iteration.
    This scan body returns one value, two policy values (left and right)
    as well as an endogenous grid value.

        Args:
            carry (tuple): The carry value passed from the previous iteration. This is a
                tuple containing the variables that are updated in each iteration.
                Including the current two optimal points j and k, the points to be saved
                this iteration as well as the indexes of the point to be inspected, the
                indicator of case 2 and the indicator if the last point was an
                intersection point.
            _iter_step (int): The current iteration number.
            value (np.ndarray): 1d array containing the unrefined value correspondence
                of shape (n_grid_wealth,).
            policy (np.ndarray): 1d array containing the unrefined policy correspondence
                of shape (n_grid_wealth,).
            endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
                grid of shape (n_grid_wealth,).
            jump_thresh (float): Jump detection threshold.
            n_points_to_scan (int): Number of points to scan in the forward and backward
                scan.

    Returns:
        tuple:

        - carry (tuple): The updated carry value passed to the next iteration.
        - result (tuple): The result of this iteration. This is a tuple containing four
            elements to be saved in this iteration:
            - value
            - left policy
            - right policy
            - endogenous grid point

    """
    (
        points_j_and_k,
        planned_to_be_saved_this_iter,
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
    ) = carry

    point_to_inspect = (
        value[idx_to_inspect],
        policy[idx_to_inspect],
        endog_grid[idx_to_inspect],
    )

    is_final_point_on_grid = idx_to_inspect == len(endog_grid) - 1

    # Conduct forward and backward scan from the point we want to inspect. We want to
    # find the point which is on the same value function segment as j. At the same time
    # we calculate the gradient from the inspected point to the respective point.
    (
        grad_next_forward,
        idx_next_on_lower_curve,
        grad_next_backward,
        idx_before_on_upper_curve,
    ) = run_forward_and_backward_scans(
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        points_j_and_k=points_j_and_k,
        idx_to_inspect=idx_to_inspect,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )

    cases, update_idx = determine_cases_and_update_idx(
        point_to_inspect=point_to_inspect,
        points_j_and_k=points_j_and_k,
        grad_next_forward=grad_next_forward,
        grad_next_backward=grad_next_backward,
        last_point_was_intersect=last_point_was_intersect,
        is_final_point_on_grid=is_final_point_on_grid,
        jump_thresh=jump_thresh,
    )

    intersection_point = select_and_calculate_intersection(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        points_j_and_k=points_j_and_k,
        idx_next_on_lower_curve=idx_next_on_lower_curve,
        idx_before_on_upper_curve=idx_before_on_upper_curve,
        idx_to_inspect=idx_to_inspect,
        case_5=cases[4],
        case_6=cases[5],
    )

    # Select the values we want to save this iteration
    point_to_save_this_iteration = select_point_to_save_this_iteration(
        intersection_point=intersection_point,
        planned_to_be_saved_this_iter=planned_to_be_saved_this_iter,
        case_6=cases[5],
    )

    point_case_2 = jax.lax.select(
        saved_last_point_already, dummy_points_grid, last_point_in_grid
    )

    point_to_be_saved_next_iteration = select_points_to_be_saved_next_iteration(
        point_to_inspect=point_to_inspect,
        point_case_2=point_case_2,
        intersection_point=intersection_point,
        planned_to_be_saved_this_iter=planned_to_be_saved_this_iter,
        cases=cases,
    )

    points_j_and_k = update_values_j_and_k(
        point_to_inspect=point_to_inspect,
        intersection_point=intersection_point,
        points_j_and_k=points_j_and_k,
        cases=cases,
    )

    (
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
    ) = update_bools_and_idx_to_inspect(
        idx_to_inspect=idx_to_inspect,
        update_idx=update_idx,
        case_2=cases[1],
        case_5=cases[4],
    )

    carry = (
        points_j_and_k,
        point_to_be_saved_next_iteration,
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
    )

    return carry, point_to_save_this_iteration


def run_forward_and_backward_scans(
    value,
    policy,
    endog_grid,
    points_j_and_k,
    idx_to_inspect,
    n_points_to_scan,
    jump_thresh,
):
    """Run the forward and backward scans at the point with idx_to_scan_from.

    We use the forward scan to find the next point that lies on the same value
    function segment as the most recent point on the upper envelope (j).
    Then we calculate the gradient between the point found on the same value function
    segment and the point we currently inspect at idx_to_scan_from.

    We use the backward scan to find the preceding point that lies on the same value
    function segment as the point we inspect. Then we calculate the gradient between
    the point found and the most recent point on the upper envelope (j).

    Args:
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        points_j_and_k (tuple): Tuple containing the value, policy and endogenous grid
            of the most recent point that lies on on the the upper envelope (j) and the
            point before (k).
        idx_to_scan_from (int): Index of the point we scan from. This should be the
            current point we inspect.
        n_points_to_scan (int): Number of points to scan.
        jump_thresh (float): Jump detection threshold.

    Returns:
        tuple:

        - grad_next_forward (float): The gradient between the next point on the same
            value function segment as j and the current point we inspect.
        - idx_next_on_lower_curve (int): Index of the next point on the same value
            function segment as j.
        - grad_next_backward (float): The gradient between the point before on the same
            value function segment as the current point we inspect and the most recent
            point that lies on the upper envelope (j).
        - idx_before_on_upper_curve (int): Index of the point before on the same value
            function segment as the current point we inspect.

    """

    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k

    (
        grad_next_forward,
        idx_next_on_lower_curve,
    ) = _forward_scan(
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        endog_grid_j=endog_grid_k_and_j[1],
        policy_j=policy_k_and_j[1],
        idx_to_inspect=idx_to_inspect,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )

    (
        grad_next_backward,
        idx_before_on_upper_curve,
    ) = _backward_scan(
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        value_j=value_k_and_j[1],
        endog_grid_j=endog_grid_k_and_j[1],
        idx_to_inspect=idx_to_inspect,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )
    return (
        grad_next_forward,
        idx_next_on_lower_curve,
        grad_next_backward,
        idx_before_on_upper_curve,
    )


def _forward_scan(
    value: jnp.ndarray,
    policy: jnp.array,
    endog_grid: jnp.ndarray,
    endog_grid_j: float,
    policy_j: float,
    idx_to_inspect: int,
    n_points_to_scan: int,
    jump_thresh: float,
) -> Tuple[float, int]:
    """Find next point on same value function as most recent point on upper envelope.

    Args:
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        endog_grid_j (float): Endogenous grid point that corresponds to the most recent
            value function point that lies on the upper envelope (j).
        policy_j (float): Point of the policy function that corresponds to the most
            recent value function point that lies on the upper envelope (j).
        idx_to_scan_from (int): Index of the point we want to scan from. This should
            be the current point we inspect.
        n_points_to_scan (int): Number of points to scan.
        jump_thresh (float): Threshold for the jump in the value function.

    Returns:
        tuple:

        - grad_next_forward (float): The gradient of the next point on the same
            value function.
        - idx_on_same_value (int): Index of next point on the value function.

    """
    indexes_to_scan = idx_to_inspect + np.arange(1, n_points_to_scan + 1, dtype=int)
    (
        grad_next_on_same_value,
        idx_on_same_value,
    ) = back_and_forward_scan_wrapper(
        endog_grid_to_calculate_gradient=endog_grid[idx_to_inspect],
        value_to_calculate_gradient=value[idx_to_inspect],
        endog_grid_to_scan_from=endog_grid_j,
        policy_to_scan_from=policy_j,
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        indexes_to_scan=indexes_to_scan,
        jump_thresh=jump_thresh,
    )

    return (
        grad_next_on_same_value,
        idx_on_same_value,
    )


def _backward_scan(
    value: jnp.ndarray,
    policy: jnp.array,
    endog_grid: jnp.ndarray,
    endog_grid_j,
    value_j,
    idx_to_inspect: int,
    n_points_to_scan: int,
    jump_thresh: float,
) -> Tuple[float, int]:
    """Find previous point on same value function as idx_to_scan_from.

    Args:
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        endog_grid_j (float): Endogenous grid point that corresponds to the most recent
            value function point that lies on the upper envelope (j).
        value_j (float): Point of the value function that corresponds to the most recent
            value function point that lies on the upper envelope (j).
        idx_to_scan_from (int): Index of the point we want to scan from. This should
            be the current point we inspect.
        n_points_to_scan (int): Number of points to scan.
        jump_thresh (float): Threshold for the jump in the value function.

    Returns:
        tuple:

        - grad_before_on_same_value (float): The gradient of the previous point on
            the same value function.
        - is_before_on_same_value (int): Indicator for whether we have found a
            previous point on the same value function.

    """
    indexes_to_scan = idx_to_inspect - np.arange(1, n_points_to_scan + 1, dtype=int)
    (
        grad_before_on_same_value,
        idx_point_before_on_same_value,
    ) = back_and_forward_scan_wrapper(
        endog_grid_to_calculate_gradient=endog_grid_j,
        value_to_calculate_gradient=value_j,
        endog_grid_to_scan_from=endog_grid[idx_to_inspect],
        policy_to_scan_from=policy[idx_to_inspect],
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        indexes_to_scan=indexes_to_scan,
        jump_thresh=jump_thresh,
    )

    return grad_before_on_same_value, idx_point_before_on_same_value


def back_and_forward_scan_wrapper(
    endog_grid_to_calculate_gradient,
    value_to_calculate_gradient,
    endog_grid_to_scan_from,
    policy_to_scan_from,
    endog_grid,
    value,
    policy,
    indexes_to_scan,
    jump_thresh,
):
    """Wrapper function to execute the backwards and forward scan.

    Args:
        endog_grid_to_calculate_gradient (float): The endogenous grid point to calculate
            the gradient from.
        value_to_calculate_gradient (float): The value function point to calculate the
            gradient from.
        endog_grid_to_scan_from (float): The endogenous grid point to scan from. We want
            to find the grid point which is on the same value function segment as the
            point we scan from.
        policy_to_scan_from (float): The policy function point to scan from. We want to
            find the grid point which is on the same value function segment as the point
            we scan from.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        indexes_to_scan (np.ndarray): 1d array of shape (n_points_to_scan,) containing
            the indexes to scan.
        jump_thresh (float): Threshold for the jump in the value function.


    Returns:
        tuple:

        - grad_we_search_for (float): The gradient we search for.
        - idx_on_same_value (int): The index of the point on the same value function
            segment as the point we scan from.

    """
    # Prepare body function by partialing in, everything except carry and counter
    partial_body = partial(
        back_and_forward_scan_body,
        endog_grid_to_calculate_gradient=endog_grid_to_calculate_gradient,
        value_to_calculate_gradient=value_to_calculate_gradient,
        endog_grid_to_scan_from=endog_grid_to_scan_from,
        policy_to_scan_from=policy_to_scan_from,
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        jump_thresh=jump_thresh,
    )

    # Initialize starting values
    found_value_already = False
    idx_on_same_value = 0
    grad_we_search_for = 0.0

    # These values will be updated each iteration.
    carry_to_update = (
        found_value_already,
        idx_on_same_value,
        grad_we_search_for,
    )

    # Execute scan function. The result is the final carry value.
    final_carry, _ = jax.lax.scan(
        f=partial_body, init=carry_to_update, xs=indexes_to_scan
    )

    # Read out final carry.
    (
        found_value_already,
        idx_on_same_value,
        grad_we_search_for,
    ) = final_carry

    return (
        grad_we_search_for,
        idx_on_same_value,
    )


def back_and_forward_scan_body(
    carry,
    current_scaned_index,
    endog_grid_to_calculate_gradient,
    value_to_calculate_gradient,
    endog_grid_to_scan_from,
    policy_to_scan_from,
    endog_grid,
    value,
    policy,
    jump_thresh,
):
    """The scan body to be executed at each iteration of the backwards and forward scan
    function.

    Args:
        carry (tuple): The carry value passed from the previous iteration. This is a
            tuple containing the variables that are updated in each iteration.
        current_scaned_index (int): The current index to be scanned.
        endog_grid_to_calculate_gradient (float): The endogenous grid point to calculate
            the gradient from.
        value_to_calculate_gradient (float): The value function point to calculate the
            gradient from.
        endog_grid_to_scan_from (float): The endogenous grid point to scan from. We want
            to find the grid point which is on the same value function segment as the
            point we scan from.
        policy_to_scan_from (float): The policy function point to scan from. We want to
            find the grid point which is on the same value function segment as the point
            we scan from.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        jump_thresh (float): Threshold for the jump in the value function.

    Returns:
        tuple:

        - carry (tuple): The updated carry value passed to the next iteration.
        - None: Dummy value to be returned.

    """
    (
        found_value_already,
        idx_on_same_value,
        grad_we_search_for,
    ) = carry

    # Make sure that we do not index out of bounds. If we reach the beginning or end of
    # the array, we'll just du dummy evaluations on the same point.
    max_index_in_array = endog_grid.shape[0] - 1
    idx_scan = jax.lax.clamp(0, current_scaned_index, max_index_in_array)

    is_not_on_same_value = create_indicator_if_value_function_is_switched(
        endog_grid_1=endog_grid_to_scan_from,
        policy_1=policy_to_scan_from,
        endog_grid_2=endog_grid[idx_scan],
        policy_2=policy[idx_scan],
        jump_thresh=jump_thresh,
    )
    is_on_same_value = ~is_not_on_same_value

    grad_to_idx_to_scan = calc_gradient(
        x1=endog_grid_to_calculate_gradient,
        y1=value_to_calculate_gradient,
        x2=endog_grid[idx_scan],
        y2=value[idx_scan],
    )
    # Now check if this is the first value on the same value function
    # This is only 1 if so far there hasn't been found a point and the point is on
    # the same value function
    is_point_we_search = is_on_same_value & (1 - found_value_already)
    # Update if you have found a point. Always 1 (=True) if you have found a point
    # already
    found_value_already = found_value_already | is_on_same_value

    # Update the first time a new point is found
    idx_on_same_value += idx_scan * is_point_we_search

    # Update the first time a new point is found
    grad_we_search_for += grad_to_idx_to_scan * is_point_we_search
    return (
        found_value_already,
        idx_on_same_value,
        grad_we_search_for,
    ), None


def update_bools_and_idx_to_inspect(idx_to_inspect, update_idx, case_2, case_5):
    """Update indicators and index of the point to be inspected in the next period.

    The indicators are booleans that capture cases where
    - we have saved the last point we checked already, and
    - the last point we inspected is an intersection point.

    Args:
        idx_to_inspect (int): Index of the point to be inspected in the current
            iteration.
        update_idx (bool): Indicator if the index should be updated.
        case_2 (bool): Indicator if we have reached the final point on the grid.
        case_5 (bool): Indicator if we are in the situation where we added the
            intersection point this iteration and add the point to inspect in the
            next iteration.

    Returns:
        tuple:

        - idx_to_inspect (int): Index of the point to inspect in the next
            iteration.
        - saved_last_point_already (bool): Indicator if we have saved the previous
            point already.
        - last_point_was_intersect (bool): Indicator if the most recent point was an
            intersection point.

    """
    idx_to_inspect += update_idx

    # In the iteration where case_2 is True for the first time, the last point we
    # checked is selected and afterwards only nans are added.
    saved_last_point_already = case_2
    last_point_was_intersect = case_5

    return idx_to_inspect, saved_last_point_already, last_point_was_intersect


def update_values_j_and_k(point_to_inspect, intersection_point, points_j_and_k, cases):
    """Update point j and k, i.e. the two most recent points on the upper envelope.

    Args:
        point_to_inspect (tuple): Tuple containing the value, policy, and
            endogenous grid point of to the point to be inspected.
        intersection_point (tuple): Tuple containing the value, policy, and endogenous
            grid points of the intersection point.
        points_j_and_k (tuple): Tuple containing the value, policy, and endogenous grid
            points of the most recent point on the upper envelope (j) and the
            point before (k).
        cases (tuple): Tuple containing the indicators for the different cases.

    Returns:
        tuple:

        - points_j_and_k (tuple): Tuple containing the updated value, policy, and
            endogenous grid point of the most recent point on the upper envelope (j)
            and the point before (k).

    """
    (
        intersect_grid,
        intersect_value,
        _intersect_policy_left,
        intersect_policy_right,
    ) = intersection_point

    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k
    value_to_inspect, policy_to_inspect, endog_grid_to_inspect = point_to_inspect

    case_1, case_2, case_3, case_4, case_5, case_6 = cases
    in_case_123 = case_1 | case_2 | case_3
    in_case_1236 = case_1 | case_2 | case_3 | case_6
    in_case_45 = case_4 | case_5

    # In case 1, 2, 3 the old value remains as value_j, in 4, 5, value_j is former
    # value k and in 6 the old value_j is overwritten
    value_j_new = (
        in_case_123 * value_k_and_j[1]
        + in_case_45 * value_to_inspect
        + case_6 * intersect_value
    )
    value_k_new = in_case_1236 * value_k_and_j[0] + in_case_45 * value_k_and_j[1]

    value_k_and_j = value_k_new, value_j_new
    policy_j_new = (
        in_case_123 * policy_k_and_j[1]
        + in_case_45 * policy_to_inspect
        + case_6 * intersect_policy_right
    )
    policy_k_new = in_case_1236 * policy_k_and_j[0] + in_case_45 * policy_k_and_j[1]
    policy_k_and_j = policy_k_new, policy_j_new
    endog_grid_j_new = (
        in_case_123 * endog_grid_k_and_j[1]
        + in_case_45 * endog_grid_to_inspect
        + case_6 * intersect_grid
    )
    endog_grid_k_new = (
        in_case_1236 * endog_grid_k_and_j[0] + in_case_45 * endog_grid_k_and_j[1]
    )
    endog_grid_k_and_j = endog_grid_k_new, endog_grid_j_new

    return value_k_and_j, policy_k_and_j, endog_grid_k_and_j


def select_points_to_be_saved_next_iteration(
    point_to_inspect,
    point_case_2,
    intersection_point,
    planned_to_be_saved_this_iter,
    cases,
):
    """Select the points to be saved in the next iteration, depending on the case.

    Args:
        point_to_inspect (tuple): Tuple containing the value, policy, and endogenous
            grid points of the point to be inspected.
        point_case_2 (jnp.ndarray): Tuple containing the value, policy, and endogenous
            grid points of the point to be saved in case 2.
        intersection_point (tuple): Tuple containing the value, policy, and endogenous
            grid points of the intersection point.
        planned_to_be_saved_this_iter (tuple): Tuple containing the value, policy, and
            endogenous grid point of the point to be saved in this iteration.
        cases (tuple): Tuple containing the indicators for the different cases.

    Returns:
        tuple:

        - point_to_be_saved_next_iteration (tuple): Tuple containing the value,
            policy left, policy right and endogenous grid point of the point to be
            saved in the next iteration.

    """
    case_1, case_2, case_3, case_4, case_5, case_6 = cases
    value_case_2, policy_case_2, endog_grid_case_2 = point_case_2
    value_to_inspect, policy_to_inspect, endog_grid_to_inspect = point_to_inspect

    (
        planned_value,
        planned_policy_left,
        planned_policy_right,
        planned_endog_grid,
    ) = planned_to_be_saved_this_iter

    (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    ) = intersection_point

    in_case_146 = case_1 | case_4 | case_6

    value_to_be_saved_next = (
        in_case_146 * value_to_inspect
        + case_2 * value_case_2
        + case_5 * intersect_value
        + case_3 * planned_value
    )
    policy_left_to_be_saved_next = (
        in_case_146 * policy_to_inspect
        + case_2 * policy_case_2
        + case_5 * intersect_policy_left
        + case_3 * planned_policy_left
    )
    policy_right_to_be_saved_next = (
        in_case_146 * policy_to_inspect
        + case_2 * policy_case_2
        + case_5 * intersect_policy_right
        + case_3 * planned_policy_right
    )
    endog_grid_to_be_saved_next = (
        in_case_146 * endog_grid_to_inspect
        + case_2 * endog_grid_case_2
        + case_5 * intersect_grid
        + case_3 * planned_endog_grid
    )

    return (
        value_to_be_saved_next,
        policy_left_to_be_saved_next,
        policy_right_to_be_saved_next,
        endog_grid_to_be_saved_next,
    )


def determine_cases_and_update_idx(
    point_to_inspect,
    points_j_and_k,
    grad_next_forward,
    grad_next_backward,
    last_point_was_intersect,
    is_final_point_on_grid,
    jump_thresh,
):
    """Determine cases and if the index to be scanned this iteration should be updated.

    This function is crucial for the optimality of the FUES. We want to have a clear
    documentation of how the cases determined here map into the into the situations
    on how the candidate solutions after solving the euler equation can look like.
    We need to do more work here!

    Args:
        point_to_inspect (tuple): Tuple containing the value, policy and endogenous grid
            of the point to be inspected.
        points_j_and_k (tuple): Tuple containing the value, policy, and endogenous grid
            point of the most recent point that lies on the upper envelope (j) and
            the point before (k).
        grad_next_forward (float): The gradient between the next point on the same
            value function segment as j and the current point we inspect.
        grad_next_backward (float): The gradient between the point before on the same
            value function segment as the current point we inspect and the most recent
            point that lies on the upper envelope (j).
        last_point_was_intersect (bool): Indicator if the last point was an
            intersection point.
        is_final_point_on_grid (bool): Indicator if this is the final point on the grid.
        jump_thresh (float): Jump detection threshold.

    Returns:
        tuple:

        - cases (tuple): Tuple containing the indicators for the different cases.
        - update_idx (bool): Indicator if the index should be updated.

    """
    value_k_and_j, _policy_k_and_j, endog_grid_k_and_j = points_j_and_k
    value_to_inspect, _policy_to_inspect, endog_grid_to_inspect = point_to_inspect

    grad_before = calc_gradient(
        x1=endog_grid_k_and_j[1],
        y1=value_k_and_j[1],
        x2=endog_grid_k_and_j[0],
        y2=value_k_and_j[0],
    )

    # gradient with leading index to be checked
    grad_next = calc_gradient(
        x1=endog_grid_to_inspect,
        y1=value_to_inspect,
        x2=endog_grid_k_and_j[1],
        y2=value_k_and_j[1],
    )

    suboptimal_cond, does_the_value_func_switch = check_for_suboptimality(
        points_j_and_k=points_j_and_k,
        point_to_inspect=point_to_inspect,
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
    case_2 = is_final_point_on_grid & ~case_1
    case_3 = suboptimal_cond & ~case_1 & ~case_2
    case_4 = ~does_the_value_func_switch * ~case_1 * ~case_2 * ~case_3
    case_5 = next_point_past_intersect & ~case_1 & ~case_2 & ~case_3 & ~case_4
    case_6 = point_j_past_intersect & ~case_1 & ~case_2 & ~case_3 & ~case_4 & ~case_5

    in_case_134 = case_1 | case_3 | case_4
    update_idx = in_case_134 | (~in_case_134 & suboptimal_cond)

    return (case_1, case_2, case_3, case_4, case_5, case_6), update_idx


def check_for_suboptimality(
    points_j_and_k,
    point_to_inspect,
    grad_next,
    grad_next_forward,
    grad_before,
    jump_thresh,
):
    """Check if current point is sub-optimal.

    Even if the function returns False, the point may still be suboptimal.
    That is iff, in the next iteration, we find that this point actually lies after a
    switching point.

    Here, we check if the point fulfills one of three conditions:
    1) Either the value function is decreasing with decreasing value function,
    2) the value function is not montone in consumption, or
    3) if the gradient of the index we inspect and the point j (the most recent point
        on the same choice-specific policy) is shallower than the gradient joining
        the i+1 and j. If True, delete the j'th point.

    If the point to inspect is the same as point j, this is always false and we
    switch the value function as well. Therefore, the third if is chosen.

    Args:
        points_j_and_k (tuple): Tuple containing the value, policym and endogenous grid
            point of the most recent point on the upper envelope (j) and the point
            before (k).
        point_to_inspect (tuple): Tuple containing the value, policy, and endogenous
            grid point of the point we inspect.
        grad_next (float): The gradient between the most recent point that lies on the
            upper envelope (j) and the point we inspect.
        grad_next_forward (float): The gradient between the next point on the same value
            function segment as j and the current point we inspect.
        grad_before (float): The gradient between the most recent point on the upper
            envelope (j) and the point before (k).
        jump_thresh (float): Jump detection threshold.

    Returns:
        tuple:

        - suboptimal_cond (bool): Indicator if the point is suboptimal.
        - does_the_value_func_switch (bool): Indicator if the value function switches.

    """
    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k
    value_to_inspect, policy_to_inspect, endog_grid_to_inspect = point_to_inspect

    does_the_value_func_switch = create_indicator_if_value_function_is_switched(
        endog_grid_1=endog_grid_k_and_j[1],
        policy_1=policy_k_and_j[1],
        endog_grid_2=endog_grid_to_inspect,
        policy_2=policy_to_inspect,
        jump_thresh=jump_thresh,
    )
    switch_value_func_and_steep_increase_after = (
        grad_next < grad_next_forward
    ) & does_the_value_func_switch

    decreasing_value = value_to_inspect < value_k_and_j[1]

    are_savings_non_monotone = check_for_non_monotone_savings(
        endog_grid_j=endog_grid_k_and_j[1],
        policy_j=policy_k_and_j[1],
        endog_grid_idx_to_inspect=endog_grid_to_inspect,
        policy_idx_to_inspect=policy_to_inspect,
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
    """Check if savings are a non-monotone in wealth.

    Check the grid between the most recent point on the upper envelope (j)
    and the current point we inspect.

    Args:
        endog_grid_j (float): The endogenous grid point of the most recent point
            on the upper envelope.
        policy_j (float): The value of the policy function of the most recent point
            on the upper envelope.
        endog_grid_idx_to_inspect (float): The endogenous grid point of the point we
            check.
        policy_idx_to_inspect (float): The policy index of the point we check.

    Returns:
        non_monotone_policy (bool): Indicator if the policy is non-monotone in wealth
            between the most recent point on the upper envelope and the point we check.

    """
    exog_grid_j = endog_grid_j - policy_j
    exog_grid_idx_to_inspect = endog_grid_idx_to_inspect - policy_idx_to_inspect
    are_savings_non_monotone = exog_grid_idx_to_inspect < exog_grid_j

    return are_savings_non_monotone


def select_point_to_save_this_iteration(
    intersection_point,
    planned_to_be_saved_this_iter,
    case_6,
):
    """Select point on the value function to be saved, depending on the case.

    This is the point which, in the previous iteration, was marked to be saved
    this iteration. Except in case 6, where we realize that this point actually
    needs to be disregarded.

    Args:
        intersection_point (tuple): Tuple containing the value, policy, and endogenous
            grid point of the intersection point.
        planned_to_be_saved_this_iter (tuple): Tuple containing the value, policy, and
            endogenous grid point of the point to be saved this iteration.
        case_6 (bool): Indicator if we are in case 6.


    Returns:
        tuple:

        - point_to_save_this_iteration (tuple): Tuple containing the value, policy left,
            policy right, and endogenous grid point of the point to be saved this
            iteration.

    """
    (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    ) = intersection_point
    (
        planned_value,
        planned_policy_left,
        planned_policy_right,
        planned_endog_grid,
    ) = planned_to_be_saved_this_iter

    # Determine variables to save this iteration. This is always the variables
    # carried from last iteration. Except in case 6.
    value_to_save = planned_value * (1 - case_6) + intersect_value * case_6
    policy_left_to_save = (
        planned_policy_left * (1 - case_6) + intersect_policy_left * case_6
    )
    policy_right_to_save = (
        planned_policy_right * (1 - case_6) + intersect_policy_right * case_6
    )
    endog_grid_to_save = planned_endog_grid * (1 - case_6) + intersect_grid * case_6

    return value_to_save, policy_left_to_save, policy_right_to_save, endog_grid_to_save


def select_and_calculate_intersection(
    endog_grid,
    policy,
    value,
    points_j_and_k,
    idx_next_on_lower_curve,
    idx_before_on_upper_curve,
    idx_to_inspect,
    case_5,
    case_6,
):
    """Select points we use to compute the intersection points.

    This functions maps very nicely into Figure 5 of the paper.
    In case 5, we use the next point (q in the graph) we found on the value function
    segment of point j (i in graph) and intersect it with the idx_to_check
    (i + 1 in the graph).
    In case 6, we are in the situation on the right-hand side in Figure 5.
    Here, we intersect the line of j and k (i and i-1 in the graph) with the line of
    idx_to_check (i+1 in the graph) and the point before on the same value function
    segment.

    Args:
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        points_j_and_k (tuple): A tuple containing the value, policy, and endogenous
            wealth grid point of the two most recent points on the upper envelope.
        idx_next_on_lower_curve (int): The index of the next point on the lower curve.
        idx_before_on_upper_curve (int): The index of the point before on the upper
            curve.
        idx_to_inspect (int): The index of the point to inspect.
        case_5 (bool): Indicator if we are in case 5.
        case_6 (bool): Indicator if we are in case 6.

    Returns:
        intersect_grid (float): The endogenous grid point of the intersection point.
        intersect_value (float): The value of the value function at the intersection.
        intersect_policy_left (float): The value of the left-continuous policy function
            at the intersection point.
        intersect_policy_right (float): The value of the right-continuous policy
            function at the intersection point.

    """
    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k

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


def create_indicator_if_value_function_is_switched(
    endog_grid_1: float,
    policy_1: float,
    endog_grid_2: float,
    policy_2: float,
    jump_thresh: float,
):
    """Create boolean to indicate whether value function switches between two points.

    Args:
        endog_grid_1 (float): The first endogenous wealth grid point.
        policy_1 (float): The policy function at the first endogenous wealth grid point.
        endog_grid_2 (float): The second endogenous wealth grid point.
        policy_2 (float): The policy function at the second endogenous wealth grid
            point.
        jump_thresh (float): Jump detection threshold.

    Returns:
        bool: Indicator if value function is switched.

    """
    exog_grid_1 = endog_grid_1 - policy_1
    exog_grid_2 = endog_grid_2 - policy_2
    gradient_exog_grid = calc_gradient(
        x1=endog_grid_1,
        y1=exog_grid_1,
        x2=endog_grid_2,
        y2=exog_grid_2,
    )
    gradient_exog_abs = jnp.abs(gradient_exog_grid)
    is_switched = gradient_exog_abs > jump_thresh

    return is_switched
