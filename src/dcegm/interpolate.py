from typing import Callable

import numpy as np
from scipy import interpolate


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

    interpolation_func = interpolate.interp1d(
        x=policy[0, :],
        y=policy[1, :],
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )

    policy_interp = interpolation_func(flat_wealth)

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

    # Calculate t+1 value function in non-constrained region
    # via inter- and extrapolation
    interpolation_func = interpolate.interp1d(
        x=value[0, :],  # endogenous wealth grid
        y=value[1, :],  # value_function
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )
    value_interp[~credit_constrained_region] = interpolation_func(
        flat_wealth[~credit_constrained_region]
    )

    return value_interp
