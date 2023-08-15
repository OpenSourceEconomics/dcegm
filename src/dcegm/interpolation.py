from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import vmap


def interpolate_and_calc_marginal_utilities(
    compute_marginal_utility: Callable,
    compute_value: Callable,
    choice: int,
    next_period_wealth: jnp.ndarray,
    endog_grid_child_state_choice: jnp.array,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
):
    """Interpolate marginal utilities.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
        next_period_wealth (jnp.ndarray): The agent's next period wealth.
            Array of shape (n_quad_stochastic, n_grid_wealth,).
        choice (int): Discrete choice of an agent.
        endog_grid_child_state_choice (jnp.ndarray): 1d array containing the endogenous
            wealth grid of the child state/choice pair. Shape (n_grid_wealth,).
        choice_policies_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding policy function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        choice_values_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding value function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).

    Returns:
        tuple:

        - marg_utils (float): Interpolated marginal utility function.
        - value_interp (float): Interpolated value function.

    """
    ind_high, ind_low = get_index_high_and_low(
        x=endog_grid_child_state_choice, x_new=next_period_wealth
    )
    marg_utils, value_interp = vmap(
        vmap(
            calc_interpolated_values_and_marg_utils,
            in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None),
    )(
        policy_child_state_choice[ind_high],
        value_child_state_choice[ind_high],
        endog_grid_child_state_choice[ind_high],
        policy_child_state_choice[ind_low],
        value_child_state_choice[ind_low],
        endog_grid_child_state_choice[ind_low],
        next_period_wealth,
        compute_value,
        compute_marginal_utility,
        endog_grid_child_state_choice[1],
        value_child_state_choice[0],
        choice,
    )

    return marg_utils, value_interp


def calc_interpolated_values_and_marg_utils(
    policy_high: float,
    value_high: float,
    wealth_high: float,
    policy_low: float,
    value_low: float,
    wealth_low: float,
    new_wealth: float,
    compute_value: Callable,
    compute_marginal_utility: Callable,
    endog_grid_min: float,
    value_min: float,
    choice: int,
):
    """Calculate interpolated marginal utility and value function.
    Args:
        policy_high (float): Policy function value at the higher end of the
            interpolation interval.
        value_high (float): Value function value at the higher end of the
            interpolation interval.
        wealth_high (float): Endogenous wealth grid value at the higher end of the
            interpolation interval.
        policy_low (float): Policy function value at the lower end of the
            interpolation interval.
        value_low (float): Value function value at the lower end of the
            interpolation interval.
        wealth_low (float): Endogenous wealth grid value at the lower end of the
            interpolation interval.
        new_wealth (float): New endogenous wealth grid value.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
        compute_marginal_utility (callable): Function for calculating the marginal
            utility from consumption level. The input ```params``` is already
            partialled in.
        endog_grid_min (float): Minimum endogenous wealth grid value.
        value_min (float): Minimum value function value.
        choice (int): Discrete choice of an agent.

    Returns:
        tuple:

        - marg_util_interp (float): Interpolated marginal utility function.
        - value_interp (float): Interpolated value function.

    """

    policy_interp, value_interp_on_grid = interpolate_policy_and_value(
        policy_high=policy_high,
        value_high=value_high,
        wealth_high=wealth_high,
        policy_low=policy_low,
        value_low=value_low,
        wealth_low=wealth_low,
        wealth_new=new_wealth,
    )

    value_interp_closed_form = compute_value(
        consumption=new_wealth, next_period_value=value_min, choice=choice
    )

    credit_constraint = new_wealth < endog_grid_min
    value_interp = (
        credit_constraint * value_interp_closed_form
        + (1 - credit_constraint) * value_interp_on_grid
    )

    marg_utility_interp = compute_marginal_utility(policy_interp)

    return marg_utility_interp, value_interp


def interpolate_policy_and_value(
    policy_high: float,
    value_high: float,
    wealth_high: float,
    policy_low: float,
    value_low: float,
    wealth_low: float,
    wealth_new: float,
):
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


def interpolate_policy_and_value_on_wealth_grid(
    begin_of_period_wealth: jnp.ndarray,
    endog_wealth_grid: jnp.array,
    policy_left_grid: jnp.ndarray,
    policy_right_grid: jnp.ndarray,
    value_grid: jnp.ndarray,
):
    """Interpolate policy and value functions on the wealth grid.

    Args:
        begin_of_period_wealth (jnp.ndarray): 1d array of shape (n,) containing the
            begin of period wealth.
        endog_wealth_grid (jnp.array): 1d array of shape (n,) containing the endogenous
            wealth grid.
        policy_grid (jnp.ndarray): 1d array of shape (n,) containing the policy function
            values corresponding to the endogenous wealth grid.
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
        x=endog_wealth_grid, x_new=begin_of_period_wealth
    )

    policy_new, value_new = interpolate_policy_and_value(
        policy_high=jnp.take(policy_left_grid, ind_high),
        value_high=jnp.take(value_grid, ind_high),
        wealth_high=jnp.take(endog_wealth_grid, ind_high),
        policy_low=jnp.take(policy_right_grid, ind_low),
        value_low=jnp.take(value_grid, ind_low),
        wealth_low=jnp.take(endog_wealth_grid, ind_low),
        wealth_new=begin_of_period_wealth,
    )

    return policy_new, value_new


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
