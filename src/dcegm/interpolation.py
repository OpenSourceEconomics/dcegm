from typing import Callable
from typing import Dict
from typing import Tuple

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

    # Now recalculate the value when consumed all wealth
    utility = compute_utility(
        consumption=new_wealth,
        params=params,
        **state_choice_vec,
    )
    value_interp_closed_form = utility + params["beta"] * value_at_zero_wealth

    # Check if we are in the credit constrained region
    credit_constraint = new_wealth <= endog_grid_min

    # If so we return the value if all is consumed.
    value_interp = (
        credit_constraint * value_interp_closed_form
        + (1 - credit_constraint) * value_interp_on_grid
    )

    return value_interp


def interp_value_and_policy_on_wealth(
    wealth: float | jnp.ndarray,
    policy: float | jnp.ndarray,
    value: float | jnp.ndarray,
    endog_grid: float | jnp.ndarray,
    compute_utility: Callable,
    endog_grid_min: float | jnp.ndarray,
    value_at_zero_wealth: float | jnp.ndarray,
    state_choice_vec: Dict[str, int],
    params: Dict[str, float],
) -> Tuple[float, float]:
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
        wealth (float): New endogenous wealth grid value.
        compute_marginal_utility (callable): Function for calculating the marginal
            utility from consumption level. The input ```params``` is already
            partialled in.
        endog_grid_min (float): Minimum endogenous wealth grid value.
        value_min (float): Minimum value function value.
        choice (int): Discrete choice of an agent.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - marg_util_interp (float): Interpolated marginal utility function.
        - value_interp (float): Interpolated value function.

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
        endog_grid_min=endog_grid_min,
        value_at_zero_wealth=value_at_zero_wealth,
        state_choice_vec=state_choice_vec,
        params=params,
    )

    return policy_interp, value_interp
