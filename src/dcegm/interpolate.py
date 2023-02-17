from functools import partial
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap


# @partial(jit, static_argnums=(0,))
def interpolate_and_calc_marginal_utilities(
    compute_marginal_utility: Callable,
    next_period_wealth: jnp.ndarray,
    choice_policies_child: jnp.ndarray,
):
    """Interpolate marginal utilities.

    Returns:
        Callable: Interpolated marginal utility function.

    """
    policy_child_state_choice_specific = vmap(interpolate_policy, in_axes=(None, 0))(
        next_period_wealth, choice_policies_child
    )

    marg_utilities_child_state_choice_specific = vmap(
        compute_marginal_utility, in_axes=0
    )(policy_child_state_choice_specific)
    return marg_utilities_child_state_choice_specific


# @jit
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
    policy_interp = linear_interpolation_with_extrapolation_jax(
        x=policy[0, :], y=policy[1, :], x_new=flat_wealth
    )
    return policy_interp


# @partial(jit, static_argnums=(3,))
def interpolate_value(
    wealth: float,
    value: jnp.ndarray,
    choice: int,
    compute_value: Callable,
) -> np.ndarray:
    """Interpolate the agent's value for given flat wealth matrix.

    Args:
        wealth (np.ndarray): Flat array of shape
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

    # Calculate t+1 value function in constrained region using
    # the analytical part
    value_interp_calc = compute_value(
        wealth,
        next_period_value=value[1, 0],
        choice=choice,
    )

    value_interp_interpol = linear_interpolation_with_extrapolation_jax(
        x=value[0, :], y=value[1, :], x_new=wealth
    )
    indicator_constrained = (wealth < value[0, 1]).astype(int)

    value_final = jnp.take(
        jnp.append(value_interp_interpol, value_interp_calc), indicator_constrained
    )

    return value_final


# @jit
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
    # ToDo: Is x after the envelope monotone?
    ind = jnp.argsort(x)
    x = jnp.take(x, ind)
    y = jnp.take(y, ind)

    ind_high = jnp.searchsorted(x, x_new).clip(max=(x.shape[0] - 1), min=1)
    ind_high -= jnp.isnan(x[ind_high]).astype(int)

    ind_low = ind_high - 1

    y_high = jnp.take(y, ind_high)
    y_low = jnp.take(y, ind_low)
    x_high = jnp.take(x, ind_high)
    x_low = jnp.take(x, ind_low)

    interpolate_dist = x_new - x_low
    interpolate_slope = (y_high - y_low) / (x_high - x_low)
    interpol_res = (interpolate_slope * interpolate_dist) + y_low

    return interpol_res


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
