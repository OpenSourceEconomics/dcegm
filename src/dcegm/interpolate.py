from typing import Callable

import numpy as np


def interpolate_policy(flat_wealth: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """Interpolate the agent's policy for given flat wealth matrix.

    Args:
        flat_wealth (np.ndarray): Flat array of shape
            (n_quad_stochastic * n_grid_wealth,) containing the agent's
            potential wealth matrix in given period.
        policy (np.ndarray): Policy array of shape (2, 1.1 * n_grid_wealth).
            Position [0, :] of the arrays contain the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the (consumption) policy
            function c(M, d), for each time period and each discrete choice.

    Returns:
        np.ndarray: Interpolated flat policy function of shape
            (n_quad_stochastic * n_grid_wealth,).
    """
    policy = policy[:, ~np.isnan(policy).any(axis=0)]
    policy_interp = linear_interpolation_with_extrapolation(
        x=policy[0, :], y=policy[1, :], x_new=flat_wealth
    )
    return policy_interp


def interpolate_value(
    flat_wealth: np.ndarray,
    value: np.ndarray,
    choice: int,
    compute_value: Callable,
) -> np.ndarray:
    """Interpolate the agent's value for given flat wealth matrix.

    Args:
        flat_wealth (np.ndarray): Flat array of shape
            (n_quad_stochastic * n_grid_wealth,) containing the agent's
            potential wealth matrix in given period.
        value (np.ndarray): Value array of shape (2, 1.1 * n_grid_wealth).
            Position [0, :] of the array contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.


    Returns:
        np.ndarray: Interpolated flat value function of shape
            (n_quad_stochastic * n_grid_wealth,).

    """
    value = value[:, ~np.isnan(value).any(axis=0)]
    value_interp = np.empty(flat_wealth.shape)

    credit_constrained_region = flat_wealth < value[0, 1]

    # Calculate t+1 value function in constrained region using
    # the analytical part
    value_interp[credit_constrained_region] = compute_value(
        flat_wealth[credit_constrained_region],
        next_period_value=value[1, 0],
        choice=choice,
    )

    value_interp[~credit_constrained_region] = linear_interpolation_with_extrapolation(
        x=value[0, :], y=value[1, :], x_new=flat_wealth[~credit_constrained_region]
    )

    return value_interp


def linear_interpolation_with_extrapolation(x, y, x_new):
    ind_high = np.searchsorted(x, x_new).clip(max=(x.shape[0] - 1), min=1)
    ind_low = ind_high - 1

    y_high = y[ind_high]
    y_low = y[ind_low]
    x_high = x[ind_high]
    x_low = x[ind_low]

    interpolate_dist = x_new - x_low
    interpolate_slope = (y_high - y_low) / (x_high - x_low)
    interpol_res = (interpolate_slope * interpolate_dist) + y_low

    return interpol_res
