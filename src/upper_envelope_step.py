"""Implementation of the Upper Envelope algorithm."""
import copy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.optimize import brenth as root

eps = 2.2204e-16


def call_upper_envelope_step(
    policy: List[np.ndarray],
    value: List[np.ndarray],
    expected_value: np.ndarray,
    period: int,
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_func: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calls the upper envelope algorithm and drops sub-optimal points.

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
    worker’s anticipation of landing exactly at the kink points in the
    subsequent periods t + 1, t + 2, ..., T under the optimal consumption policy.

    
    Args:
        policy (List(np.ndarray)): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            concurrent local optima for consumption. In the case where the consumption
            policy function has no discontinuities, i.e. only one solution to the 
            Euler equation exists, we have *n_endog_wealth_grid* = n_grid_wealth + 1.
        value (List(np.ndarray)): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions. In the case where the value function
            has no non-concavities, we have *n_endog_wealth_grid* = n_grid_wealth + 1.
        period (int): Current period t.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_func (callable): The agent's utility function.
        
    
    Returns:
        (tuple) Tuple containing:
        
        - policy_refined (np.ndarray): Worker's *refined* (consumption) policy 
            function of the current period, where suboptimal points have been dropped 
            and the kink points along with the corresponding interpolated values of 
            the policy function have been added. Shape (2, *n_grid_refined*), where 
            *n_grid_refined* is the length of the *refined* endogenous wealth grid.

        - value_refined (np.ndarray): Worker's *refined* value function of the 
            current period, where suboptimal points have been dropped and the kink 
            points along with the corresponding interpolated values of the value 
            function have been added. Shape (2, *n_grid_refined*), where 
            *n_grid_refined* is the length of the *refined* endogenous wealth grid.
    """
    policy = copy.deepcopy(policy[period][1])  # state == 1 "working"
    value = copy.deepcopy(value[period][1])

    min_wealth_grid = np.min(value[0, 1:])

    if value[0, 1] <= min_wealth_grid:
        (
            value_refined,
            points_to_add,
            index_dominated_points,
        ) = locate_non_concave_region_and_refine(value)
    else:
        # Non-concave region coincides with credit constraint.
        # This happens when we have a non-monotonicity in the endogenous wealth grid
        # that goes below the first point.
        # Solution: Value function to the left of the first point is analytical,
        # so we just need to add some points to the left of the first grid point.
        policy, value = _augment_grid(
            policy,
            value,
            expected_value,
            min_wealth_grid,
            params,
            options,
            utility_func,
        )
        (
            value_refined,
            points_to_add,
            index_dominated_points,
        ) = locate_non_concave_region_and_refine(value)
        value_refined = np.hstack([np.array([[0], [expected_value[0]]]), value_refined])

    if len(index_dominated_points) > 0:
        policy_refined = refine_policy(policy, index_dominated_points, points_to_add)
    else:
        policy_refined = policy

    return policy_refined, value_refined


def locate_non_concave_region_and_refine(
    value,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locates non-concave regions and refines the value "correspondence".

    Non-concave regions in the value function are reflected by non-monotonous
    regions in the underlying endogenous wealth grid.

    Multiple solutions to the Euler equation cause the standard EGM loop to
    produce a “value correspondence” rather than a value function. 
    The elimination of suboptimal grid points converts this correspondence back
    to a proper function.
    
    
    Args:
        value (np.ndarray):  Array storing the choice-specific value function
            "correspondences". Shape (2, *n_endog_wealth_grid*), where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions in the value function. 
            In the presence of kinks, the value function is a "correspondence" 
            rather than a function due to non-concavities.
    
    Returns:
        (tuple) Tuple containing:

        - value_refined (np.ndarray): Array of shape (2, *n_grid_refined*)
            containing the *refined* choice-specific value functions, which means that 
            suboptimal points have been removed from the endogenous wealth grid and
            the value function "correspondence". Furthermore, kink points and the 
            corresponding interpolated values of the value function have been added.

        - points_to_add (np.ndarray): Array of shape (*n_kink_points*,) 
            containing the kink points and corresponding interpolated values of the 
            *refined* value function that have been added to ``value_refined``.
            *n_kink_points* is of variable length.
    
        - index_dominated_points (np.ndarray): Array of shape (*n_dominated_points*,) 
            containing the indices of dominated points in the endogenous wealth grid,
            where *n_dominated_points* is of variable length.
    """
    value_correspondence = copy.deepcopy(value)
    segments_non_mono = []

    # Find non-monotonicity in the endogenous wealth grid where grid point
    # to the right that is smaller than the preceeding one.
    is_monotonic = value_correspondence[0, 1:] > value_correspondence[0, :-1]

    lap = 1
    move_right = True

    while move_right:
        index_non_monotonic = np.where(is_monotonic != is_monotonic[0])[0]

        if len(index_non_monotonic) == 0:

            # Check if we are beyond the starting (left-most) point
            if lap > 1:
                segments_non_mono += [value_correspondence]
            move_right = False

        else:
            index_non_monotonic = min(index_non_monotonic)  # left-most point

            part_one, part_two = _partition_grid(
                value_correspondence, index_non_monotonic
            )
            segments_non_mono += [part_one]
            value_correspondence = part_two

            # Move point of first non-monotonicity to the right
            is_monotonic = is_monotonic[index_non_monotonic:]

            lap += 1

    if len(segments_non_mono) > 1:
        segments_non_mono = [np.sort(i) for i in segments_non_mono]
        value_refined, points_to_add = compute_upper_envelope(segments_non_mono,)
        index_dominated_points = _find_dominated_points(value, value_refined, 10)
    else:
        value_refined = value
        points_to_add = np.stack([np.array([]), np.array([])])
        index_dominated_points = np.array([])

    return value_refined, points_to_add, index_dominated_points


def compute_upper_envelope(segments: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes upper envelope and refines value function "correspondence".

    The upper envelope algorithm detects suboptimal points in the value function
    "correspondence. Consequently, (i) the suboptimal points are removed and the
    (ii) kink points along with their corresponding interpolated values are included.
    The elimination of suboptimal grid points converts the value
    "correspondence" back to a proper function. Applying both (i) and (ii)
    yields the *refined* endogenous wealth grid and the *refined* value function.
    
    Args:
        segments (List[np.ndarray]): List of non-monotonous segments in the
            endogenous wealth grid, which results in non-concavities in the 
            corresponding value function. The list contains n_non_monotonous 
            np.ndarrays of shape (2, *len_non_monotonous*), where 
            *len_non_monotonous* is of variable length denoting the length of the 
            given non-monotonous segment.

    Returns:
        (tuple) Tuple containing:

        - points_upper_env_refined (np.ndarray): Array containing the *refined*
            endogenous wealth grid and the corresponding value function. 
            *refined* means suboptimal points have been dropped and the kink points
            along with the corresponding interpolated values of the value function
            have beend added.
            Shape (2, *n_grid_refined*), where *n_grid_refined* is the length of
            the *refined* endogenous grid.

        - points_to_add (np.ndarray): Array containing the kink points and 
            corresponding interpolated values of the value function that have been 
            added to ``points_upper_env_refined``.
            Shape (2, *n_intersect_points*), where *n_intersect_points* is the number of
            intersection points between the two uppermost segments 
            (i.e. ``first_segment`` and ``second_segment``).
    """
    endog_wealth_grid = np.unique(
        np.concatenate([segments[arr][0].tolist() for arr in range(len(segments))])
    )

    values_interp = np.empty((len(segments), len(endog_wealth_grid)))
    for arr in range(len(segments)):
        values_interp[arr, :] = _get_interpolated_value(
            segments, index=arr, grid_points=endog_wealth_grid, fill_value_=-np.inf
        )

    max_values_interp = np.tile(values_interp.max(axis=0), (3, 1))  # need this below
    top_segments = values_interp == max_values_interp[0, :]

    grid_points_upper_env = [endog_wealth_grid[0]]
    values_upper_env = [values_interp[0, 0]]
    intersect_points_upper_env = []
    values_intersect_upper_env = []

    # Index of top segment, starting at first (left-most) grid point
    index_first_segment = np.where(top_segments[:, 0] == 1)[0][0]

    move_right = True

    while move_right:
        index_first_segment = np.where(top_segments[:, 0] == 1)[0][0]

        for i in range(1, len(endog_wealth_grid)):
            index_second_segment = np.where(top_segments[:, i] == 1)[0][0]

            if index_second_segment != index_first_segment:
                first_segment = index_first_segment
                second_segment = index_second_segment
                first_grid_point = endog_wealth_grid[i - 1]
                second_grid_point = endog_wealth_grid[i]

                values_first_segment = _get_interpolated_value(
                    segments,
                    index=first_segment,
                    grid_points=[first_grid_point, second_grid_point],
                )
                values_second_segment = _get_interpolated_value(
                    segments,
                    index=second_segment,
                    grid_points=[first_grid_point, second_grid_point],
                )

                if np.all(
                    np.isfinite(np.stack([values_first_segment, values_second_segment]))
                ) and np.all(np.abs(values_first_segment - values_second_segment) > 0):
                    intersect_point = root(
                        _subtract_values,
                        first_grid_point,
                        second_grid_point,
                        args=(segments[first_segment], segments[second_segment],),
                    )
                    value_intersect = _get_interpolated_value(
                        segments, index=first_segment, grid_points=intersect_point,
                    )

                    values_all_segments = np.empty((len(segments), 1))
                    for segment in range(len(segments)):
                        values_all_segments[segment] = _get_interpolated_value(
                            segments,
                            index=segment,
                            grid_points=intersect_point,
                            fill_value_=-np.inf,
                        )

                    index_max_value_intersect = np.where(
                        values_all_segments == values_all_segments.max(axis=0)
                    )[0][0]

                    if (index_max_value_intersect == first_segment) | (
                        index_max_value_intersect == second_segment
                    ):
                        # There are no other functions above
                        grid_points_upper_env.append(intersect_point)
                        values_upper_env.append(value_intersect)

                        intersect_points_upper_env.append(intersect_point)
                        values_intersect_upper_env.append(value_intersect)

                        if second_segment == index_second_segment:
                            move_right = False
                        else:
                            first_segment = second_segment
                            first_grid_point = intersect_point
                            second_segment = index_second_segment
                            second_grid_point = endog_wealth_grid[i]
                    else:
                        second_segment = index_max_value_intersect
                        second_grid_point = intersect_point

            # Add point if it lies currently on the highest segment
            if (
                any(abs(segments[index_second_segment][0] - endog_wealth_grid[i]) < eps)
                is True
            ):
                grid_points_upper_env.append(endog_wealth_grid[i])
                values_upper_env.append(max_values_interp[0, i])

            index_first_segment = index_second_segment

        points_upper_env_refined = np.empty((2, len(grid_points_upper_env)))
        points_upper_env_refined[0, :] = grid_points_upper_env
        points_upper_env_refined[1, :] = values_upper_env

        points_to_add = np.empty((2, len(intersect_points_upper_env)))
        points_to_add[0] = intersect_points_upper_env
        points_to_add[1] = values_intersect_upper_env

    return points_upper_env_refined, points_to_add


def refine_policy(
    policy: np.ndarray, index_dominated_points: np.ndarray, points_to_add: np.ndarray
) -> np.ndarray:
    """Drops suboptimal points from policy "correspondence" and adds new optimal ones.
    
    Args:
        points_to_add (np.ndarray): Array of shape (*n_kink_points*,),
            containing the kink points and corresponding interpolated values of 
            the refined value function, where *n_kink_points* is of variable
            length.
        index_dominated_points (np.ndarray): Array of shape (*n_dominated_points*,) 
            containing the indices of dominated points in the endogenous wealth grid,
            where *n_dominated_points* is of variable length.

    Returns:
        policy_refined (np.ndarray): Array of shape (2, *n_grid_refined*)
            containing the *refined* choice-specific policy function, which means that 
            suboptimal points have been removed from the endogenous wealth grid and
            the policy "correspondence". Furthermore, kink points and the 
            corresponding interpolated values of the policy function have been added.
    """
    # Remove suboptimal consumption points
    endog_wealth_grid = np.delete(policy[0, :], index_dominated_points)
    optimal_consumption = np.delete(policy[1, :], index_dominated_points)

    # Add new optimal consumption points
    new_points_policy_interp = []

    for new_grid_point in range(len(points_to_add[0, :])):
        all_points_to_the_left = np.where(
            policy[0, :] < points_to_add[0, new_grid_point]
        )[0]
        all_points_to_the_right = np.where(
            policy[0, :] > points_to_add[0, new_grid_point]
        )[0]

        last_point_to_the_left = max(
            all_points_to_the_left[
                ~np.isin(all_points_to_the_left, index_dominated_points)
            ]
        )

        # Find (scalar) point interpolated from the left
        interpolation_left = interpolate.interp1d(
            policy[0, :][last_point_to_the_left : last_point_to_the_left + 2],
            policy[1, :][last_point_to_the_left : last_point_to_the_left + 2],
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_from_the_left = interpolation_left(
            points_to_add[0][new_grid_point]
        )  # single point

        first_point_to_the_right = min(
            all_points_to_the_right[
                ~np.isin(all_points_to_the_right, index_dominated_points)
            ]
        )

        # Find (scalar) point interpolated from the right
        interpolation_right = interpolate.interp1d(
            policy[0, :][first_point_to_the_right - 1 : first_point_to_the_right + 1],
            policy[1, :][first_point_to_the_right - 1 : first_point_to_the_right + 1],
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_from_the_right = interpolation_right(points_to_add[0, new_grid_point])

        new_points_policy_interp += [
            np.array(
                [
                    points_to_add[0, new_grid_point],
                    interp_from_the_left,
                    interp_from_the_right,
                ]
            )
        ]

    # Insert new points into the endogenous wealth grid and consumption policy
    for to_add in range(len(new_points_policy_interp)):
        index_insert = np.where(
            endog_wealth_grid > new_points_policy_interp[to_add][0]
        )[0][0]

        # 1) Add new points to policy TWICE to accurately describe discontinuities
        endog_wealth_grid = np.insert(
            endog_wealth_grid, index_insert, new_points_policy_interp[to_add][0],
        )
        endog_wealth_grid = np.insert(
            endog_wealth_grid,
            index_insert + 1,
            new_points_policy_interp[to_add][0] - 0.001 * 2.2204e-16,
        )

        # 2a) Add new optimal consumption point, interpolated from the left
        optimal_consumption = np.insert(
            optimal_consumption, index_insert, new_points_policy_interp[to_add][1],
        )
        # 2b) Add new optimal consumption point, interpolated from the right
        optimal_consumption = np.insert(
            optimal_consumption, index_insert + 1, new_points_policy_interp[to_add][2],
        )

    policy_refined = np.stack([endog_wealth_grid, optimal_consumption])

    # Make sure first element in endogenous wealth grid and optiomal consumption policy
    # are both 0.
    if policy_refined[0, 0] != 0.0:
        policy_refined = np.hstack([np.zeros((2, 1)), policy_refined])

    return policy_refined


def _augment_grid(
    policy: np.ndarray,
    value: np.ndarray,
    expected_value: np.ndarray,
    min_wealth_grid: float,
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_func: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extends the endogenous wealth grid, value, and policy function to the left.
    
    Args:
        policy (np.ndarray):  Array storing the choice-specific 
            policy function. Shape (2, *n_endog_wealth_grid*), where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            discontinuities in the policy function. 
            In the presence of discontinuities, the policy function is a 
            "correspondence" rather than a function due to multiple local optima.
        value (np.ndarray):  Array storing the choice-specific 
            value function. Shape (2, *n_endog_wealth_grid*), where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions in the value function. 
            In the presence of kinks, the value function is a "correspondence" 
            rather than a function due to non-concavities.
        expected_value (np.ndarray): Array of current period's expected value of
            next_period. Shape (*n_endog_wealth_grid*,).
        min_wealth_grid (float): Minimal point in the endogenous wealth grid.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_func (callable): The agent's utility function. 

    Returns:
        policy_augmented (np.ndarray): Array containing endogenous grid and 
            policy function with ancillary points added to the left. 
            Shape (2, *n_grid_augmented*).
        value_augmented (np.ndarray): Array containing endogenous grid and 
            value function with ancillary points added to the left. 
            Shape (2, *n_grid_augmented*).
    """
    delta = params.loc[("delta", "delta"), "value"]  # disutility of work
    beta = params.loc[("beta", "beta"), "value"]  # discount factor
    n_grid_wealth = options["grid_points_wealth"]

    grid_points_to_add = np.linspace(
        min_wealth_grid, value[0, 1], int(np.floor(n_grid_wealth / 10))
    )[:-1]

    values_to_add = (
        utility_func(grid_points_to_add, params) - delta + beta * expected_value[0]
    )
    value_augmented = np.stack(
        [
            np.append(grid_points_to_add, value[0, 1:]),
            np.append(values_to_add, value[1, 1:]),
        ]
    )
    policy_augmented = np.stack(
        [
            np.append(grid_points_to_add, policy[0, 1:]),
            np.append(grid_points_to_add, policy[1, 1:]),
        ]
    )

    return policy_augmented, value_augmented


def _partition_grid(
    value_correspondence: np.ndarray, j: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits the grid into two parts, 1,..., j and j, j+1,..., J.

    Note that the index ``j``, after which the separation occurs,
    is also included in the second parition.

    Args:
        value_correspondence (np.ndarray):  Array storing the choice-specific 
            value function "correspondences". Shape (2, *n_endog_wealth_grid*), where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions in the value function. 
            In the presence of kinks, the value function is a "correspondence" 
            rather than a function due to non-concavities.
        j (int): Index denoting the location where the endogenous wealth grid is
            separated.

    Returns:
        part_one (np.ndarray): Array of shape (2, : ``j`` + 1) containing the first 
            partition, where
        part_two (np.ndarray): Array of shape (2, ``j``:) containing the second partition.
    """
    j = value_correspondence.shape[1] if j > value_correspondence.shape[1] else j
    part_one = np.stack(
        [
            value_correspondence[0, : j + 1],  # endogenous wealth grid
            value_correspondence[1, : j + 1],  # corresponding value function
        ]
    )

    # Include boundary points in both partitions
    part_two = np.stack([value_correspondence[0, j:], value_correspondence[1, j:]])

    return part_one, part_two


def _find_dominated_points(
    value_correspondence: np.ndarray, value_refined: np.ndarray, significance: int = 10,
) -> np.ndarray:
    """Returns indexes of dominated points in the value function "correspondence".

    Equality is measured up to 10**(-``significance``).

    Args:
        value_correspondence (np.ndarray):  Array storing the choice-specific 
            value function "correspondences". Shape (2, *n_endog_wealth_grid*), where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions in the value function. 
            In the presence of kinks, the value function is a "correspondence" 
            rather than a function due to non-concavities.
        value_refined (np.ndarray): Array of *refined* value function, where
            suboptimal points have been dropped and kink points along with the
            corresponding interpolated values of the value function have been added.
            Shape (2, *n_grid_refined*), where *n_grid_refined* is the length of
            the *refined* endogenous grid.
        significance (float): Level of significance. Equality is measured up to 
            10**(-``significance``).

    Returns:
        index_dominated_points (np.ndarray): Array of shape (*n_dominated_points*,) 
            containing the indices of dominated points in the endogenous wealth grid,
            where *n_dominated_points* is of variable length. 
    """
    sig_ = 10 ** significance
    _sig = 10 ** (-significance)

    # Endogenous wealth grid
    correspond_grid = np.round(value_correspondence[0, :] * sig_) * _sig
    refined_grid = np.round(value_refined[0, :] * sig_) * _sig

    # Value function
    correspond_value = np.round(value_correspondence[1, :] * sig_) * _sig
    refined_value = np.round(value_refined[1, :] * sig_) * _sig

    index_all = np.arange(len(correspond_grid))
    index_dominated_points = np.union1d(
        index_all[~np.isin(correspond_grid, refined_grid)],
        index_all[~np.isin(correspond_value, refined_value)],
    )

    return index_dominated_points


def _get_interpolated_value(
    segments: List[np.ndarray],
    index: int,
    grid_points: Union[float, List[float]],
    fill_value_: Any = np.nan,
) -> Tuple[Union[np.ndarray, float]]:
    """Returns the intepolated value(s)."""
    interp_func = interpolate.interp1d(
        segments[index][0],
        segments[index][1],
        bounds_error=False,
        fill_value=fill_value_,
    )
    values_interp = interp_func(grid_points)

    return values_interp


def _subtract_values(grid_point: float, first_segment, second_segment):
    """Subtracts the interpolated values of the two uppermost segments."""
    first_interp_func = interpolate.interp1d(
        first_segment[0], first_segment[1], bounds_error=False, fill_value="extrapolate"
    )
    second_interp_func = interpolate.interp1d(
        second_segment[0],
        second_segment[1],
        bounds_error=False,
        fill_value="extrapolate",
    )
    values_first_segment = first_interp_func(grid_point)
    values_second_segment = second_interp_func(grid_point)

    diff_values_segments = values_first_segment - values_second_segment

    return diff_values_segments
