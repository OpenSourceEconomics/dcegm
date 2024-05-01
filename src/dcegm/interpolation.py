from typing import Callable
from typing import Dict

import jax.numpy as jnp
from jax import numpy as jnp


def linear_interpolation_formula(
    y_high: float | jnp.ndarray,
    y_low: float | jnp.ndarray,
    x_high: float | jnp.ndarray,
    x_low: float | jnp.ndarray,
    x_new: float | jnp.ndarray,
):
    """Linear interpolation formula."""
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


def interp_value_and_check_creditconstraint(
    value_high: float | jnp.ndarray,
    wealth_high: float | jnp.ndarray,
    value_low: float | jnp.ndarray,
    wealth_low: float | jnp.ndarray,
    new_wealth: float | jnp.ndarray,
    compute_utility: Callable,
    endog_grid_min: float,
    value_at_zero_wealth: float,
    state_choice_vec: Dict[str, int],
    params: Dict[str, float],
) -> float | jnp.ndarray:
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
        endog_grid_min (float): Minimum endogenous wealth grid value.
        value_min (float): Minimum value function value.
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - marg_util_interp (float): Interpolated marginal utility function.
        - value_interp (float): Interpolated value function.

    """

    value_interp_on_grid = linear_interpolation_formula(
        y_high=value_high,
        y_low=value_low,
        x_high=wealth_high,
        x_low=wealth_low,
        x_new=new_wealth,
    )

    value_interp = check_value_if_credit_constrained(
        value_interp_on_grid=value_interp_on_grid,
        value_at_zero_wealth=value_at_zero_wealth,
        new_wealth=new_wealth,
        endog_grid_min=endog_grid_min,
        params=params,
        state_choice_vec=state_choice_vec,
        compute_utility=compute_utility,
    )
    return value_interp


def check_value_if_credit_constrained(
    value_interp_on_grid,
    value_at_zero_wealth,
    new_wealth,
    endog_grid_min,
    params,
    state_choice_vec,
    compute_utility,
):
    """This function takes the value interpolated on the solution and checks if it is in
    the region, where consume all your wealth is the optimal solution.

    This is by construction endog_grid_min. If so, it returns the closed form solution
    for the value function, by calculating the utility of consuming all the wealth and
    adding the discounted expected value of zero wealth. Otherwise, it returns the
    interpolated value function.

    """
    utility = compute_utility(
        consumption=new_wealth,
        params=params,
        **state_choice_vec,
    )
    value_interp_closed_form = utility + params["beta"] * value_at_zero_wealth

    credit_constraint = new_wealth <= endog_grid_min
    value_interp = (
        credit_constraint * value_interp_closed_form
        + (1 - credit_constraint) * value_interp_on_grid
    )
    return value_interp
