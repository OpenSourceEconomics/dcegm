"""Implementation of the Fast Upper-Envelope Scan.

Based on Akshay Shanker, University of Sydney, akshay.shanker@me.com.

"""
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from dcegm.interpolate import linear_interpolation_with_extrapolation


def fast_upper_envelope_wrapper_org(
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

    # TODO: determine locations where enogenous grid points are # noqa: T000
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

    value_clean_with_nans = _scan_org(
        endog_grid, value, policy, exog_grid, m_bar=jump_thresh, LB=10
    )
    # _scan_org(e_grid, vf, c, a_prime, m_bar, LB, fwd_scan_do=True)

    endog_grid_refined = (endog_grid[np.where(~np.isnan(value_clean_with_nans))],)
    value_refined = (value_clean_with_nans[np.where(~np.isnan(value_clean_with_nans))],)
    policy_refined = (policy[np.where(~np.isnan(value_clean_with_nans))],)

    return endog_grid_refined, value_refined, policy_refined


# ================================================================================

# from numba import njit


# @njit
def append_push(x_array, m):
    """Delete first value of array, pushes back index of all undeleted values and
    appends m to final index."""

    for i in range(len(x_array) - 1):
        x_array[i] = x_array[i + 1]

    x_array[-1] = m
    return x_array


# @njit
def back_scan_gradients(m_array, a_prime, vf_full, e_grid, j, q):
    """Compute gradients of value correspondence points and policy points with respect
    to all m values and policy points in m_array See Figure 5, right panel in DS
    (2023)"""

    gradients_m_vf = np.zeros(len(m_array))
    gradients_m_a = np.zeros(len(m_array))

    for m in range(len(gradients_m_a)):
        m_int = int(m_array[m])
        gradients_m_vf[m] = (vf_full[j] - vf_full[m_int]) / (e_grid[j] - e_grid[m_int])
        gradients_m_a[m] = np.abs(
            (a_prime[q] - a_prime[m_int]) / (e_grid[q] - e_grid[m_int])
        )

    return gradients_m_vf, gradients_m_a


# @njit
def fwd_scan_gradients(a_prime, vf_full, e_grid, j, q, LB):
    """Computes gradients of value correspondence points and  policy points with respect
    to values and policy points for next LB points in grid See Figure 5, left panel in
    DS (2023)"""

    gradients_f_vf = np.zeros(LB)
    gradients_f_a = np.zeros(LB)

    for f in range(LB):
        gradients_f_vf[f] = (vf_full[q] - vf_full[q + 1 + f]) / (
            e_grid[q] - e_grid[q + 1 + f]
        )
        gradients_f_a[f] = np.abs(
            (a_prime[j] - a_prime[q + 1 + f]) / (e_grid[j] - e_grid[q + 1 + f])
        )

    return gradients_f_vf, gradients_f_a


# @njit
def perp(a):
    """Finds perpendicilar line to 1D line
    Parameters
    ----------
    a: 1D array
        points (b, 1/m)
    Returns
    -------
    b: 1D array
        b[0] = -1/m, b[1]= b
    """
    b = np.empty(np.shape(a))
    b[0] = -a[1]
    b[1] = a[0]

    return b


# @njit
def seg_intersect(a1, a2, b1, b2):
    """Intersection of two 1D line segments
    Parameters
    ----------
    a1: 1D array
         First point of first line seg
    a2: 1D array
         Second point of first line seg
    b1: 1D array
         First point of first line seg
    b2: 1D array
         Second point of first line seg
    Returns
    -------
    c: 1D array
        intersection point
    """
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom) * db + b1


# @njit
def FUES(e_grid, vf, c, a_prime, b=1e-10, m_bar=2, LB=10):
    """
    FUES function returns refined EGM grid, value function and
    policy function
    Parameters
    ----------
    e_grid: 1D array
            unrefined endogenous grid
    vf: 1D array
            value correspondence points at endogenous grid
    c: 1D array
            policy 1 points at endogenous grid
    a_prime: 1D array
            policy 2 points at endogenous grid
    b: float64
        lower bound for the endogenous grid
    m_bar: float64
            jump detection threshold
    LB: int
         length of bwd and fwd scan search
    Returns
    -------
    e_grid_clean: 1D array
                    refined endogenous grid
    vf_clean: 1D array
                value function on refined grid
    c_clean: 1D array
                policy 1 on refined grid
    a_prime_clean: 1D array
                    policy 2 on refined grid
    dela: 1D array
            gradient of policy 2 on refined grid
    Notes
    -----
    Policy 2 is used to determine jumps in policy.
    FUES attaches NaN values to vf array
    where points are sub-optimal and not to be retained.
    The code below checks to see if multiple EGM points equal the lower
    bound of the endogenous grid. If multiple EGM points equal the lower bound,
    the one yielding the highest value is retained. So far in applications
    in DS (2023),the only multiple EGM values occur on the
    lower bound (see Application 2 for DS, 2023).
    Todo
    ----
    Incorporate explicit check for multiple
    equal EGM grid values (other than the lb).
    Incorporate full functionality to attach crossing points.
    """

    # determine locations where enogenous grid points are
    # equal to the lower bound
    if len(vf[np.where(e_grid <= b)]) > 0:
        vf_lb_max = max(vf[np.where(e_grid <= b)])

        # remove sub-optimal lb EGM points
        for i in range(len(e_grid)):
            if e_grid[i] <= b and vf[i] < vf_lb_max:
                vf[i] = np.nan

    # remove NaN values from vf array
    e_grid = e_grid[np.where(~np.isnan(vf))]
    c = c[np.where(~np.isnan(vf))]
    a_prime = a_prime[np.where(~np.isnan(vf))]
    vf = vf[np.where(~np.isnan(vf))]

    # sort policy and vf by e_grid order
    vf = np.take(vf, np.argsort(e_grid))
    c = np.take(c, np.argsort(e_grid))
    a_prime = np.take(a_prime, np.argsort(e_grid))
    e_grid = np.sort(e_grid)

    # scan attaches NaN to vf at all sub-optimal points
    e_grid_clean, vf_with_nans, c_clean, a_prime_clean, dela = _scan(
        e_grid, vf, c, a_prime, m_bar, LB
    )

    return (
        e_grid_clean[np.where(~np.isnan(vf_with_nans))],
        vf[np.where(~np.isnan(vf_with_nans))],
        c_clean[np.where(~np.isnan(vf_with_nans))],
        a_prime_clean[np.where(~np.isnan(vf_with_nans))],
        dela,
    )


# @njit
def _scan_org(e_grid, vf, c, a_prime, m_bar, LB, fwd_scan_do=True):
    """ " Implements the scan for FUES."""

    # leading index for optimal values j
    # leading index for value to be `checked' is i+1

    # create copy of value function
    # this copy remains intact as the unrefined set of points
    vf_full = np.copy(vf)

    # empty array to store policy function gradient
    dela = np.zeros(len(vf))

    # array of previously sub-optimal indices to be used in backward scan
    m_array = np.zeros(LB)

    # FUES scan
    for i in range(len(e_grid) - 2):

        # inital two points are optimal (assumption)
        if i <= 1:
            j = np.copy(np.array([i]))[0]
            k = np.copy(np.array([j - 1]))[0]
            previous_opt_is_intersect = False
            k_minus_1 = np.copy(np.array([k]))[0] - 1

        else:
            # value function gradient betweeen previous two optimal points
            g_j_minus_1 = (vf_full[j] - vf_full[k]) / (e_grid[j] - e_grid[k])

            # gradient with leading index to be checked
            g_1 = (vf_full[i + 1] - vf_full[j]) / (e_grid[i + 1] - e_grid[j])

            # policy gradient with leading index to be checked
            g_tilde_a = np.abs(
                (a_prime[i + 1] - a_prime[j]) / (e_grid[i + 1] - e_grid[j])
            )

            # if right turn is made and jump registered
            # remove point or perform forward scan
            if g_1 < g_j_minus_1 and g_tilde_a > m_bar:
                keep_i_1_point = False

                if fwd_scan_do:
                    gradients_f_vf, gradients_f_a = fwd_scan_gradients(
                        a_prime, vf_full, e_grid, j, i + 1, LB
                    )

                    # get index of closest next point with same
                    # discrete choice as point j
                    if len(np.where(gradients_f_a < m_bar)[0]) > 0:
                        m_index_fwd = np.where(gradients_f_a < m_bar)[0][0]
                        g_m_vf = gradients_f_vf[m_index_fwd]
                        g_m_a = gradients_f_a[m_index_fwd]

                        if g_1 > g_m_vf:
                            keep_i_1_point = True
                        else:
                            pass
                    else:
                        pass

                    if not keep_i_1_point:
                        vf[i + 1] = np.nan
                        m_array = append_push(m_array, i + 1)
                    else:
                        previous_opt_is_intersect = True
                        k = np.copy(np.array([j]))[0]
                        j = np.copy(np.array([i]))[0] + 1

            # If value falls, remove points
            elif vf_full[i + 1] - vf_full[j] < 0:
                vf[i + 1] = np.nan
                # append index array of previously deleted points
                m_array = append_push(m_array, i + 1)

            # assume value is monotone in policy and delete if not
            # satisfied
            elif g_1 < g_j_minus_1 and a_prime[i + 1] - a_prime[j] < 0:
                vf[i + 1] = np.nan
                m_array = append_push(m_array, i + 1)

            # if left turn is made or right turn with no jump, then
            # keep point provisionally and conduct backward scan
            else:
                # backward scan
                # compute value gradients (from i+1) and
                # policy gradients (from j)
                # wrt to LB previously deleted values
                gradients_m_vf, gradients_m_a = back_scan_gradients(
                    m_array, a_prime, vf_full, e_grid, j, i + 1
                )
                keep_j_point = True

                # index m of last point that is deleted and does not jump from
                # leading point where left turn is made.
                # this is the closest previous point on the same
                # discrete choice specific
                # policy as the leading value we have just jumped to
                if len(np.where(gradients_m_a < m_bar)[0]) > 0:
                    m_index_bws = np.where(gradients_m_a < m_bar)[0][-1]

                    # gradient of vf and policy to the m'th point
                    g_m_vf = gradients_m_vf[m_index_bws]
                    g_m_a = gradients_m_a[m_index_bws]

                else:
                    m_index_bws = 0
                    keep_j_point = True

                # index of m'th point on the e_grid
                m_ind = int(m_array[m_index_bws])

                # if the gradient joining the leading point i+1 (we have just
                # jumped to) and the point m(the last point on the same
                # choice specific policy) is shallower than the
                # gradient joining the i+1 and j, then delete j'th point

                if g_1 > g_j_minus_1 and g_1 >= g_m_vf and g_tilde_a > m_bar:
                    keep_j_point = False

                if not keep_j_point:
                    pj = np.copy(np.array([e_grid[j], vf_full[j]]))
                    pi1 = np.copy(np.array([e_grid[i + 1], vf_full[i + 1]]))
                    pk = np.copy(np.array([e_grid[k], vf_full[k]]))
                    pm = np.copy(np.array([e_grid[m_ind], vf_full[m_ind]]))
                    intrsect = seg_intersect(pj, pk, pi1, pm)

                    vf[j] = np.nan
                    vf_full[j] = intrsect[1]
                    e_grid[j] = intrsect[0]
                    previous_opt_is_intersect = True
                    j = np.copy(np.array([i]))[0] + 1

                else:

                    previous_opt_is_intersect = False
                    if g_1 > g_j_minus_1:
                        previous_opt_is_intersect = True

                    k = np.copy(np.array([j]))[0]
                    j = np.copy(np.array([i]))[0] + 1

    return vf
