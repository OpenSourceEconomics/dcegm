from typing import Callable, Dict, Tuple

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


def interp1d_policy_and_value_on_wealth(
    wealth: float | jnp.ndarray,
    endog_grid: jnp.ndarray,
    policy: jnp.ndarray,
    value: jnp.ndarray,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    params: Dict[str, float],
    discount_factor,
) -> Tuple[float, float]:
    """Interpolate policy and value function given a single wealth grid point.

    Args:
        wealth (float | jnp.ndarray): New wealth point(s) to interpolate.
        endog_grid (jnp.ndarray): Solved endogenous wealth grid.
        policy (jnp.ndarray): Solved policy function.
        value (jnp.ndarray): Solved value function.
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        params (Dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - policy_interp (float): Interpolated optimal policy for wealth.
        - value_interp (float): Interpolated value for wealth.

    """

    # For all choices, the wealth is the same in the solution
    ind_high, ind_low = get_index_high_and_low(x=endog_grid, x_new=wealth)

    policy_interp = linear_interpolation_formula(
        y_high=policy[ind_high],
        y_low=policy[ind_low],
        x_high=endog_grid[ind_high],
        x_low=endog_grid[ind_low],
        x_new=wealth,
    )

    value_interp = interp_value_and_check_creditconstraint(
        value_high=value[ind_high],
        wealth_high=endog_grid[ind_high],
        value_low=value[ind_low],
        wealth_low=endog_grid[ind_low],
        new_wealth=wealth,
        compute_utility=compute_utility,
        endog_grid_min=endog_grid[1],
        value_at_zero_wealth=value[0],
        state_choice_vec=state_choice_vec,
        params=params,
        discount_factor=discount_factor,
    )

    return policy_interp, value_interp


def interp_value_on_wealth(
    wealth: float | jnp.ndarray,
    endog_grid: jnp.ndarray,
    value: jnp.ndarray,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    params: Dict[str, float],
    discount_factor: float,
) -> jnp.ndarray | float:
    """Interpolate value function on a single wealth point.

    Args:
        wealth (float): New wealth point to interpolate.
        endog_grid (jnp.ndarray): Solved endogenous wealth grid.
        value (jnp.ndarray): Solved value function.
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        params (Dict): Dictionary containing the model parameters.

    Returns:
        jnp.ndarray | float: Interpolated value for wealth.

    """

    ind_high, ind_low = get_index_high_and_low(x=endog_grid, x_new=wealth)

    value_interp = interp_value_and_check_creditconstraint(
        value_high=value[ind_high],
        wealth_high=endog_grid[ind_high],
        value_low=value[ind_low],
        wealth_low=endog_grid[ind_low],
        new_wealth=wealth,
        compute_utility=compute_utility,
        endog_grid_min=endog_grid[1],
        value_at_zero_wealth=value[0],
        state_choice_vec=state_choice_vec,
        params=params,
        discount_factor=discount_factor,
    )

    return value_interp


def interp_policy_on_wealth(
    wealth: float | jnp.ndarray,
    endog_grid: jnp.ndarray,
    policy: jnp.ndarray,
) -> float:
    """Interpolate policy function on a single wealth point.

    Args:
        wealth (float): New wealth point to interpolate.
        endog_grid (jnp.ndarray): Solved endogenous wealth grid.
        value (jnp.ndarray): Solved value function.
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        params (Dict): Dictionary containing the model parameters.

    Returns:
        float: Interpolated value for wealth.

    """

    ind_high, ind_low = get_index_high_and_low(x=endog_grid, x_new=wealth)

    policy_interp = linear_interpolation_formula(
        y_high=policy[ind_high],
        y_low=policy[ind_low],
        x_high=endog_grid[ind_high],
        x_low=endog_grid[ind_low],
        x_new=wealth,
    )

    return policy_interp


def interp_value_and_check_creditconstraint(
    value_high: float | jnp.ndarray,
    wealth_high: float | jnp.ndarray,
    value_low: float | jnp.ndarray,
    wealth_low: float | jnp.ndarray,
    new_wealth: float | jnp.ndarray,
    compute_utility: Callable,
    endog_grid_min: float | jnp.ndarray,
    value_at_zero_wealth: float | jnp.ndarray,
    state_choice_vec: Dict[str, int],
    params: Dict[str, float],
    discount_factor: float,
) -> float | jnp.ndarray:
    """Calculate the interpolated value with accounting for a possible credit
    constrained solution.

    This function first calculates the interpolated value and then checks if we are
    in the credit constrained region, i.e. below endog_grid_min. If so, it returns
    the value as the sum of the utility when consumed all wealth and the discounted
    value at zero savings. Creditconstrained means it is optimal to consume all!

    Args:
        value_high (float): Value function value at the higher end of the
            interpolation interval.
        wealth_high (float): Endogenous wealth grid value at the higher end of the
            interpolation interval.
        value_low (float): Value function value at the lower end of the
            interpolation interval.
        wealth_low (float): Endogenous wealth grid value at the lower end of the
            interpolation interval.
        new_wealth (float): New endogenous wealth grid value.
        endog_grid_min (float): Minimum endogenous wealth grid value.
        value_at_zero_wealth (float): The value at zero wealth.
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:
        - value_interp (float): Interpolated value function.

    """

    value_interp_on_grid = linear_interpolation_formula(
        y_high=value_high,
        y_low=value_low,
        x_high=wealth_high,
        x_low=wealth_low,
        x_new=new_wealth,
    )

    # Now recalculate the closed-form value of consuming all wealth
    utility = compute_utility(
        consumption=new_wealth,
        params=params,
        **state_choice_vec,
    )
    value_interp_closed_form = utility + discount_factor * value_at_zero_wealth

    # Check if we are in the credit constrained region
    credit_constraint = new_wealth <= endog_grid_min

    value_interp = (
        credit_constraint * value_interp_closed_form
        + (1 - credit_constraint) * value_interp_on_grid
    )

    return value_interp
