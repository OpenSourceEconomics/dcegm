from typing import Tuple

import jax.numpy as jnp
import numpy as np


def interpolate_policy_and_value(
    policy_high: float,
    value_high: float,
    wealth_high: float,
    policy_low: float,
    value_low: float,
    wealth_low: float,
    wealth_new: float,
) -> Tuple[float, float]:
    """Interpolate policy and value functions.

    Args:
        policy_high (float): Policy function value at the higher end of the
            interpolation interval.
        value_high (float): Value function value at the higher end of the
            interpolation interval.
        wealth_high (float): Wealth value at the higher end of the interpolation
            interval.
        policy_low (float): Policy function value at the lower end of the
            interpolation interval.
        value_low (float): Value function value at the lower end of the
            interpolation interval.
        wealth_low (float): Wealth value at the lower end of the interpolation
            interval.
        wealth_new (float): Wealth value at which the policy and value functions
            should be interpolated.

    Returns:
        tuple:

        - policy_new (float): Interpolated policy function value.
        - value_new (float): Interpolated value function value.

    """
    interpolate_dist = wealth_new - wealth_low
    interpolate_slope_policy = (policy_high - policy_low) / (wealth_high - wealth_low)
    interpolate_slope_value = (value_high - value_low) / (wealth_high - wealth_low)
    policy_new = (interpolate_slope_policy * interpolate_dist) + policy_low
    value_new = (interpolate_slope_value * interpolate_dist) + value_low

    return policy_new, value_new


def interpolate_policy_and_value_on_wealth_grid(
    wealth_beginning_of_period: jnp.ndarray,
    endog_wealth_grid: jnp.ndarray,
    policy_left_grid: jnp.ndarray,
    policy_right_grid: jnp.ndarray,
    value_grid: jnp.ndarray,
):
    """Interpolate policy and value functions on the wealth grid.

    This function uses the left and right policy function.
    For a more detailed description, see calc_intersection_and_extrapolate_policy
    in fast_upper_envelope.py.

    Args:
        wealth_beginning_of_period (jnp.ndarray): 1d array of shape (n,) containing the
            begin of period wealth.
        endog_wealth_grid (jnp.array): 1d array of shape (n,) containing the endogenous
            wealth grid.
        policy_left_grid (jnp.ndarray): 1d array of shape (n,) containing the
            left policy function corresponding to the endogenous wealth grid.
        policy_right_grid (jnp.ndarray): 1d array of shape (n,) containing the
            left policy function corresponding to the endogenous wealth grid.
        value_grid (jnp.ndarray): 1d array of shape (n,) containing the value function
            values corresponding to the endogenous wealth grid.

    Returns:
        tuple:

        - policy_new (jnp.ndarray): 1d array of shape (n,) containing the interpolated
            policy function values corresponding to the begin of period wealth.
        - value_new (jnp.ndarray): 1d array of shape (n,) containing the interpolated
            value function values corresponding to the begin of period wealth.

    """
    ind_high, ind_low = get_index_high_and_low(
        x=endog_wealth_grid, x_new=wealth_beginning_of_period
    )

    policy_new, value_new = interpolate_policy_and_value(
        policy_high=jnp.take(policy_left_grid, ind_high),
        value_high=jnp.take(value_grid, ind_high),
        wealth_high=jnp.take(endog_wealth_grid, ind_high),
        policy_low=jnp.take(policy_right_grid, ind_low),
        value_low=jnp.take(value_grid, ind_low),
        wealth_low=jnp.take(endog_wealth_grid, ind_low),
        wealth_new=wealth_beginning_of_period,
    )

    return policy_new, value_new


def linear_interpolation_with_extrapolation_jax(x, y, x_new):
    """Linear interpolation with extrapolation.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        y (np.ndarray): 1d array of shape (n,) containing the y-values
            corresponding to the x-values.
        x_new (float): The new x-value at which to evaluate the interpolation function.

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
    ind = np.argsort(x, kind="mergesort")
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
