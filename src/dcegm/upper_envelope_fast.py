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
    choice: int,  # noqa: 100
    n_grid_wealth: int,
    compute_value: Callable,  # noqa: 100
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
    exog_grid = np.linspace(0, max(exog_grid), len(endog_grid))

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
    value = np.take(value, np.argsort(endog_grid))
    policy = np.take(policy, np.argsort(endog_grid))
    exog_grid = np.take(exog_grid, np.argsort(endog_grid[~np.isnan(endog_grid)]))
    endog_grid = np.sort(endog_grid)

    endog_grid = endog_grid[np.where(~np.isnan(value))]
    policy = policy[np.where(~np.isnan(value))]
    exog_grid = exog_grid[np.where(~np.isnan(value))]

    value_clean_with_nans = _forward_scan(endog_grid, value, exog_grid, jump_thresh)

    endog_grid_refined = (endog_grid[np.where(~np.isnan(value_clean_with_nans))],)
    value_refined = (value_clean_with_nans[np.where(~np.isnan(value_clean_with_nans))],)
    policy_refined = (policy[np.where(~np.isnan(value_clean_with_nans))],)

    return endog_grid_refined, value_refined, policy_refined


def _forward_scan(endog_grid, value, exog_grid, jump_thresh):
    value_unrefined = np.copy(value)

    j = 2

    for i in range(2, len(endog_grid) - 2):

        current_diff = (value_unrefined[j] - value_unrefined[j - 1]) / (
            endog_grid[j] - endog_grid[j - 1]
        )

        next_diff = (value_unrefined[i + 1] - value[j]) / (
            endog_grid[i + 1] - endog_grid[j]
        )

        if np.isnan(value_unrefined[i + 1]):
            pass
        else:
            if (
                current_diff >= next_diff
                and np.abs(
                    (exog_grid[i + 1] - exog_grid[j])
                    / (endog_grid[i + 1] - endog_grid[j])
                )
                > jump_thresh
            ):
                value[i + 1] = np.nan

            else:
                j = i + 1

    return value
