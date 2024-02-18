"""Fedor's Upper Envelope algorithm.

Based on the original MATLAB code by Fedor Iskhakov:
https://github.com/fediskhakov/dcegm/blob/master/model_retirement.m

"""
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from scipy.optimize import brenth as root

from tests.utils.interpolations import linear_interpolation_with_extrapolation
from tests.utils.interpolations import (
    linear_interpolation_with_inserting_missing_values,
)

eps = 2.2204e-16


def upper_envelope(
    policy: np.ndarray,
    value: np.ndarray,
    exog_grid: np.ndarray,
    state_choice_vec: np.ndarray,
    params: Dict[str, float],
    compute_utility: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
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
        params (dict): Dictionary containing the model's parameters.
        compute_value (callable): Function to compute the agent's value.

    Returns:
        (tuple) Tuple containing
        - policy_refined (np.ndarray): Worker's *refined* (consumption) policy
            function of the current period, where suboptimal points have been dropped
            and the kink points along with the corresponding interpolated values of
            the policy function have been added. Shape (2, 1.1 * n_grid_wealth).
        - value_refined (np.ndarray): Worker's *refined* value function of the
            current period, where suboptimal points have been dropped and the kink
            points along with the corresponding interpolated values of the value
            function have been added. Shape (2, 1.1 * n_grid_wealth).

    """
    n_grid_wealth = len(exog_grid)
    min_wealth_grid = np.min(value[0, 1:])
    credit_constr = False

    if value[0, 1] <= min_wealth_grid:
        segments_non_mono = locate_non_concave_regions(value)
    else:
        # Non-concave region coincides with credit constraint.
        # This happens when there is a non-monotonicity in the endogenous wealth grid
        # that goes below the first point.
        # Solution: Value function to the left of the first point is analytical,
        # so we just need to add some points to the left of the first grid point.

        credit_constr = True
        expected_value_zero_wealth = value[1, 0]

        policy, value = _augment_grid(
            policy,
            value,
            state_choice_vec,
            expected_value_zero_wealth,
            min_wealth_grid,
            n_grid_wealth,
            params,
            compute_utility=compute_utility,
        )
        segments_non_mono = locate_non_concave_regions(value)

    if len(segments_non_mono) > 1:
        _value_refined, points_to_add = compute_upper_envelope(segments_non_mono)
        index_dominated_points = find_dominated_points(
            value, _value_refined, significance=10
        )

        if credit_constr:
            value_refined = np.hstack(
                [np.array([[0], [expected_value_zero_wealth]]), _value_refined]
            )
        else:
            value_refined = _value_refined

        policy_refined = refine_policy(policy, index_dominated_points, points_to_add)

    else:
        value_refined = value
        policy_refined = policy

    # Fill array with nans to fit 10% extra grid points,
    # as the true shape is unknown ex ante
    policy_refined_with_nans = np.empty((2, int(1.1 * n_grid_wealth)))
    value_refined_with_nans = np.empty((2, int(1.1 * n_grid_wealth)))
    policy_refined_with_nans[:] = np.nan
    value_refined_with_nans[:] = np.nan

    policy_refined_with_nans[:, : policy_refined.shape[1]] = policy_refined
    value_refined_with_nans[:, : value_refined.shape[1]] = value_refined

    return policy_refined_with_nans, value_refined_with_nans


def locate_non_concave_regions(
    value: np.ndarray,
) -> List[np.ndarray]:
    """Locates non-concave regions.
    Find non-monotonicity in the endogenous wealth grid where a grid point
    to the right is smaller than its preceding point. Put differently, the
    value function bends "backwards".
    Non-concave regions in the value function are reflected by non-monotonous
    regions in the underlying endogenous wealth grid.
    Multiple solutions to the Euler equation cause the standard EGM loop to
    produce a "value correspondence" rather than a value function.
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
    segments_non_mono = []

    is_monotonic = value[0, 1:] > value[0, :-1]

    niter = 0
    move_right = True

    while move_right:
        index_non_monotonic = np.where(is_monotonic != is_monotonic[0])[0]

        # Check if we are beyond the starting (left-most) point
        if len(index_non_monotonic) == 0:
            if niter > 0:
                segments_non_mono += [value]
            move_right = False
            break

        else:
            index_non_monotonic = min(index_non_monotonic)  # left-most point

            part_one, part_two = _partition_grid(value, index_non_monotonic)
            segments_non_mono += [part_one]
            value = part_two

            # Move point of first non-monotonicity to the right
            is_monotonic = is_monotonic[index_non_monotonic:]

            niter += 1

    return segments_non_mono


def compute_upper_envelope(
    segments: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute upper envelope and refines value function correspondence.
    The upper envelope algorithm detects suboptimal points in the value function
    correspondence. Consequently, (i) the suboptimal points are removed and the
    (ii) kink points along with their corresponding interpolated values are included.
    The elimination of suboptimal grid points converts the value
    correspondence back to a proper function. Applying both (i) and (ii)
    yields the refined endogenous wealth grid and the *refined* value function.
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
        np.concatenate([segments[arr][0] for arr in range(len(segments))])
    )

    values_interp = np.empty((len(segments), len(endog_wealth_grid)))
    for i, segment in enumerate(segments):
        values_interp[i, :] = linear_interpolation_with_inserting_missing_values(
            x=segment[0],
            y=segment[1],
            x_new=endog_wealth_grid,
            missing_value=-np.inf,
        )
    # values_interp has in each row the corresponding values of the upper curve
    # in the overlapping seg

    max_values_interp = np.tile(values_interp.max(axis=0), (3, 1))  # need this below
    top_segments = values_interp == max_values_interp[0, :]

    grid_points_upper_env = [endog_wealth_grid[0]]
    values_upper_env = [values_interp[0, 0]]
    intersect_points_upper_env = []
    values_intersect_upper_env = []

    move_right = True

    while move_right:
        # Index of top segment, starting at first (left-most) grid point
        index_first_segment = np.where(top_segments[:, 0])[0][0]

        for i in range(1, len(endog_wealth_grid)):
            index_second_segment = np.where(top_segments[:, i] == 1)[0][0]

            if index_second_segment != index_first_segment:
                first_segment = index_first_segment
                second_segment = index_second_segment
                first_grid_point = endog_wealth_grid[i - 1]
                second_grid_point = endog_wealth_grid[i]

                values_first_segment = (
                    linear_interpolation_with_inserting_missing_values(
                        x=segments[first_segment][0],
                        y=segments[first_segment][1],
                        x_new=np.array([first_grid_point, second_grid_point]),
                        missing_value=np.nan,
                    )
                )
                values_second_segment = (
                    linear_interpolation_with_inserting_missing_values(
                        x=segments[second_segment][0],
                        y=segments[second_segment][1],
                        x_new=np.array([first_grid_point, second_grid_point]),
                        missing_value=np.nan,
                    )
                )

                if np.all(
                    np.isfinite(
                        np.vstack([values_first_segment, values_second_segment])
                    )
                ) and np.all(np.abs(values_first_segment - values_second_segment) > 0):
                    intersect_point = root(
                        _subtract_values,
                        first_grid_point,
                        second_grid_point,
                        args=(
                            segments[first_segment],
                            segments[second_segment],
                        ),
                    )
                    value_intersect = (
                        linear_interpolation_with_inserting_missing_values(
                            x=segments[first_segment][0],
                            y=segments[first_segment][1],
                            x_new=np.array([intersect_point]),
                            missing_value=np.nan,
                        )[0]
                    )

                    values_all_segments = np.empty((len(segments), 1))
                    for segment in range(len(segments)):
                        values_all_segments[
                            segment
                        ] = linear_interpolation_with_inserting_missing_values(
                            x=segments[segment][0],
                            y=segments[segment][1],
                            x_new=np.array([intersect_point]),
                            missing_value=-np.inf,
                        )[
                            0
                        ]

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


def find_dominated_points(
    value_correspondence: np.ndarray,
    value_refined: np.ndarray,
    significance: int = 10,
) -> np.ndarray:
    """Returns indexes of dominated points in the value function correspondence.

    Equality is measured up to 10**(-``significance``).
    Args:
        value_correspondence (np.ndarray):  Array storing the choice-specific
            value function correspondences. Shape (2, n_endog_wealth_grid), where
            n_endog_wealth_grid is of variable length depending on the number of
            kinks and non-concave regions in the value function.
            In the presence of kinks, the value function is a correspondence
            rather than a function due to non-concavities.
        value_refined (np.ndarray): Array of refined value function, where
            suboptimal points have been dropped and kink points along with the
            corresponding interpolated values of the value function have been added.
            Shape (2, n_grid_refined), where n_grid_refined is the length of
            the refined endogenous grid.
        significance (float): Level of significance. Equality is measured up to
            10**(-``significance``).

    Returns:
        index_dominated_points (np.ndarray): Array of shape (n_dominated_points,)
            containing the indices of dominated points in the endogenous wealth grid,
            where n_dominated_points is of variable length.

    """
    sig_pos = 10**significance
    sig_neg = 10 ** (-significance)

    grid_all = np.round(value_correspondence[0, :] * sig_pos) * sig_neg
    grid_refined_sig = np.round(value_refined[0, :] * sig_pos) * sig_neg

    value_all = np.round(value_correspondence[1, :] * sig_pos) * sig_neg
    value_refined_sig = np.round(value_refined[1, :] * sig_pos) * sig_neg

    index_all = np.arange(len(grid_all))
    index_dominated_points = np.union1d(
        index_all[~np.isin(grid_all, grid_refined_sig)],
        index_all[~np.isin(value_all, value_refined_sig)],
    )

    return index_dominated_points


def refine_policy(
    policy: np.ndarray, index_dominated_points: np.ndarray, points_to_add: np.ndarray
) -> np.ndarray:
    """Drop suboptimal points from policy correspondence and add new optimal ones.

    Args:
        points_to_add (np.ndarray): Array of shape (*n_kink_points*,),
            containing the kink points and corresponding interpolated values of
            the refined value function, where *n_kink_points* is of variable
            length.
        index_dominated_points (np.ndarray): Array of shape (*n_dominated_points*,)
            containing the indices of dominated points in the endogenous wealth grid,
            where *n_dominated_points* is of variable length.

    Returns:
        (np.ndarray): Array of shape (2, *n_grid_refined*)
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
        interp_from_the_left = linear_interpolation_with_extrapolation(
            x=policy[0, :][last_point_to_the_left : last_point_to_the_left + 2],
            y=policy[1, :][last_point_to_the_left : last_point_to_the_left + 2],
            x_new=points_to_add[0][new_grid_point],
        )

        first_point_to_the_right = min(
            all_points_to_the_right[
                ~np.isin(all_points_to_the_right, index_dominated_points)
            ]
        )

        # Find (scalar) point interpolated from the right
        interp_from_the_right = linear_interpolation_with_extrapolation(
            x=policy[0, :][first_point_to_the_right - 1 : first_point_to_the_right + 1],
            y=policy[1, :][first_point_to_the_right - 1 : first_point_to_the_right + 1],
            x_new=points_to_add[0, new_grid_point],
        )

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
            endog_wealth_grid,
            index_insert,
            new_points_policy_interp[to_add][0],
        )
        endog_wealth_grid = np.insert(
            endog_wealth_grid,
            index_insert + 1,
            new_points_policy_interp[to_add][0] - 0.001 * 2.2204e-16,
        )

        # 2a) Add new optimal consumption point, interpolated from the left
        optimal_consumption = np.insert(
            optimal_consumption,
            index_insert,
            new_points_policy_interp[to_add][1],
        )
        # 2b) Add new optimal consumption point, interpolated from the right
        optimal_consumption = np.insert(
            optimal_consumption,
            index_insert + 1,
            new_points_policy_interp[to_add][2],
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
    state_choice_vec: np.ndarray,
    expected_value_zero_wealth: np.ndarray,
    min_wealth_grid: float,
    n_grid_wealth: int,
    params,
    compute_utility: Callable,
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
        expected_value_zero_wealth (float): The agent's expected value given that she
            has a wealth of zero.
        min_wealth_grid (float): Minimal wealth level in the endogenous wealth grid.
        n_grid_wealth (int): Number of grid points in the exogenous wealth grid.
        params (dict): Dictionary containing the model's parameters.
        compute_value (callable): Function to compute the agent's value.

    Returns:
        policy_augmented (np.ndarray): Array containing endogenous grid and
            policy function with ancillary points added to the left.
            Shape (2, *n_grid_augmented*).
        value_augmented (np.ndarray): Array containing endogenous grid and
            value function with ancillary points added to the left.
            Shape (2, *n_grid_augmented*).

    """

    grid_points_to_add = np.linspace(min_wealth_grid, value[0, 1], n_grid_wealth // 10)[
        :-1
    ]

    utility = compute_utility(
        consumption=grid_points_to_add,
        params=params,
        **state_choice_vec,
    )
    values_to_add = utility + params["beta"] * expected_value_zero_wealth

    value_augmented = np.vstack(
        [
            np.append(grid_points_to_add, value[0, 1:]),
            np.append(values_to_add, value[1, 1:]),
        ]
    )
    policy_augmented = np.vstack(
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
    is also included in the second partition.
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
            partition.
        part_two (np.ndarray): Array of shape (2, ``j``:) containing the second
            partition.

    """
    j = min(j, value_correspondence.shape[1])

    part_one = np.vstack(
        [
            value_correspondence[0, : j + 1],  # endogenous wealth grid
            value_correspondence[1, : j + 1],  # value function
        ]
    )

    # Include boundary points in both partitions
    part_two = np.vstack([value_correspondence[0, j:], value_correspondence[1, j:]])

    return part_one, part_two


def _subtract_values(grid_point: float, first_segment, second_segment):
    """Subtracts the interpolated values of the two uppermost segments."""
    values_first_segment = linear_interpolation_with_extrapolation(
        x=first_segment[0], y=first_segment[1], x_new=grid_point
    )
    values_second_segment = linear_interpolation_with_extrapolation(
        x=second_segment[0], y=second_segment[1], x_new=grid_point
    )

    diff_values_segments = values_first_segment - values_second_segment

    return diff_values_segments
