from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import vmap


def get_values_and_marginal_utilities(
    compute_marginal_utility: Callable,
    compute_value: Callable,
    next_period_wealth: jnp.ndarray,
    choice_policies_child: jnp.ndarray,
    value_functions_child: jnp.ndarray,
    endog_grid: jnp.array,
):
    """Interpolate marginal utilities and value functions.

    Returns:
        Callable: Interpolated marginal utility function.

    """
    full_choice_set = jnp.arange(choice_policies_child.shape[0], dtype=jnp.int32)

    marg_utilities_choice_specific, value_choice_specific = vmap(
        interpolate_and_calc_marginal_utilities, in_axes=(None, None, 0, 0, 0, None, 0)
    )(
        compute_marginal_utility,
        next_period_wealth,
        choice_policies_child,
        value_functions_child,
        endog_grid,
        compute_value,
        full_choice_set,
    )

    return marg_utilities_choice_specific, value_choice_specific


def interpolate_and_calc_marginal_utilities(
    compute_marginal_utility: Callable,
    wealth: float,
    policies: jnp.ndarray,
    value: jnp.ndarray,
    endog_grid: jnp.array,
    compute_value: Callable,
    choice: int,
):
    """Interpolate marginal utilities.
    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        wealth (np.ndarray): Flat array of shape
            (n_quad_stochastic * n_grid_wealth,) containing the agent's
            potential wealth matrix in given period.
        policies (np.ndarray): Policy array of shape (2, 1.1 * n_grid_wealth).
            Position [0, :] of the arrays contain the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the (consumption) policy
            function c(M, d), for each time period and each discrete choice.
        value (np.ndarray): Value array of shape (2, 1.1 * n_grid_wealth).
            Position [0, :] of the array contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
        choice (int): Discrete choice of an agent.

    Returns:
        float: Interpolated marginal utility function.
        float: Interpolated value function.


    """
    ind_high, ind_low = get_index_high_and_low(x=endog_grid, x_new=wealth)

    policy_interp, value_interp = interpolate_policy_and_value(
        policy_high=policies[ind_high],
        value_high=value[ind_high],
        wealth_high=endog_grid[ind_high],
        policy_low=policies[ind_low],
        value_low=value[ind_low],
        wealth_low=endog_grid[ind_low],
        wealth_new=wealth,
    )

    value_calc = compute_value(
        consumption=wealth, next_period_value=value[0], choice=choice
    )

    constraint = wealth < endog_grid[1]
    value_final = constraint * value_calc + (1 - constraint) * value_interp

    marg_utility_interp = compute_marginal_utility(policy_interp)

    return marg_utility_interp, value_final


def interpolate_policy_and_value(
    policy_high: jnp.ndarray,
    value_high: jnp.ndarray,
    wealth_high: jnp.ndarray,
    policy_low: jnp.ndarray,
    value_low: jnp.ndarray,
    wealth_low: jnp.ndarray,
    wealth_new: jnp.ndarray,
):
    """Interpolate policy and value functions.

    Args:
        policy (np.ndarray): 1d array of shape (n,) containing the policy function
            values.
        value (np.ndarray): 1d array of shape (n,) containing the value function
            values.
        endog_grid (np.ndarray): 1d array of shape (n,) containing the endogenous
            grid.
        wealth_new (np.ndarray): 1d array of shape (n,) containing the new wealth
            values at which to evaluate the interpolation function.

    Returns:
        np.ndarray: 1d array of shape (n,) containing the interpolated policy
            function values.
        np.ndarray: 1d array of shape (n,) containing the interpolated value

    """

    interpolate_dist = wealth_new - wealth_low
    interpolate_slope_policy = (policy_high - policy_low) / (wealth_high - wealth_low)
    interpolate_slope_value = (value_high - value_low) / (wealth_high - wealth_low)
    policy_new = (interpolate_slope_policy * interpolate_dist) + policy_low
    value_new = (interpolate_slope_value * interpolate_dist) + value_low

    return policy_new, value_new


def linear_interpolation_with_extrapolation_jax(x, y, x_new):
    """Linear interpolation with extrapolation.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        y (np.ndarray): 1d array of shape (n,) containing the y-values
            corresponding to the x-values.
        x_new (float): The new x-value at which to evaluate the interpolation function.
        ind_high (int): Index of the value in the wealth grid which is higher than
            x_new. Or in case of extrapolation last or first index of not nan element.

    Returns:
        float: The new y-value corresponding to the new x-value.
            In case x_new contains a value outside of the range of x, these
            values are extrapolated.

    """
    # make sure that the function also works for unsorted x-arrays
    # taken from scipy.interpolate.interp1d
    ind = jnp.argsort(x)
    x = jnp.take(x, ind)
    y = jnp.take(y, ind)

    ind_high, ind_low = get_index_high_and_low(x=x, x_new=x_new)

    y_high = jnp.take(y, ind_high)
    y_low = jnp.take(y, ind_low)
    x_high = jnp.take(x, ind_high)
    x_low = jnp.take(x, ind_low)

    interpolate_dist = x_new - x_low
    interpolate_slope = (y_high - y_low) / (x_high - x_low)
    interpol_res = (interpolate_slope * interpolate_dist) + y_low

    return interpol_res


def get_index_high_and_low(x, x_new):
    """Get index of the highest value in x that is smaller than x_new.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        x_new (float): The new x-value at which to evaluate the interpolation function.

    Returns:
        int: Index of the value in the wealth grid which is higher than x_new. Or in
            case of extrapolation last or first index of not nan element.

    """
    ind_high = jnp.searchsorted(x, x_new).clip(max=(x.shape[0] - 1), min=1)
    ind_high -= jnp.isnan(x[ind_high]).astype(int)
    return ind_high, ind_high - 1


def linear_interpolation_with_extrapolation(x, y, x_new):
    """Linear interpolation with extrapolation.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        y (np.ndarray): 1d array of shape (n,) containing the y-values
            corresponding to the x-values.
        x_new (np.ndarray or float): 1d array of shape (m,) or float containing
            the new x-values at which to evaluate the interpolation function.

    Returns:
        np.ndarray or float: 1d array of shape (m,) or float containing
            the new y-values corresponding to the new x-values.
            In case x_new contains values outside of the range of x, these
            values are extrapolated.

    """

    # make sure that the function also works for unsorted x-arrays
    # taken from scipy.interpolate.interp1d
    ind = np.argsort(x)
    x = x[ind]
    y = np.take(y, ind)

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


def linear_interpolation_with_inserting_missing_values(x, y, x_new, missing_value):
    """Linear interpolation with inserting missing values.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        y (np.ndarray): 1d array of shape (n,) containing the y-values
            corresponding to the x-values.
        x_new (np.ndarray or float): 1d array of shape (m,) or float containing
            the new x-values at which to evaluate the interpolation function.
        missing_value (np.ndarray or float): Flat array of shape (1,) or float
            to set for values of x_new outside of the range of x.

    Returns:
        np.ndarray or float: 1d array of shape (m,) or float containing the
            new y-values corresponding to the new x-values.
            In case x_new contains values outside of the range of x, these
            values are set equal to missing_value.

    """
    interpol_res = linear_interpolation_with_extrapolation(x, y, x_new)
    where_to_miss = (x_new < x.min()) | (x_new > x.max())
    interpol_res[where_to_miss] = missing_value
    return interpol_res
