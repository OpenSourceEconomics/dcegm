from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
from dcegm.interpolation import get_index_high_and_low
from dcegm.interpolation import interpolate_policy_and_value
from jax import numpy as jnp
from jax import vmap


def interpolate_value_and_calc_marginal_utility(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: np.ndarray,
    wealth_beginning_of_period: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_left_child_state_choice: jnp.ndarray,
    policy_right_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
) -> Tuple[float, float]:
    """Interpolate marginal utilities.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
        wealth_next_period (jnp.ndarray): 2d array of shape
            (n_quad_stochastic, n_grid_wealth,) containing the agent's beginning of
            period wealth.
        choice (int): The agent's discrete choice.
        endog_grid_child_state_choice (jnp.ndarray): 1d array containing the endogenous
            wealth grid of the child state/choice pair. Shape (n_grid_wealth,).
        choice_policies_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding policy function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        choice_values_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding value function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - marg_utils (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated marginal utilities for each wealth level and
            income shock.
        - value_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated value function.

    """

    ind_high, ind_low = get_index_high_and_low(
        x=endog_grid_child_state_choice, x_new=wealth_beginning_of_period
    )
    marg_utils, value_interp = vmap(
        vmap(
            _interpolate_value_and_marg_util,
            in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None),
    )(
        jnp.take(policy_left_child_state_choice, ind_high),
        value_child_state_choice[ind_high],
        endog_grid_child_state_choice[ind_high],
        jnp.take(policy_right_child_state_choice, ind_low),
        value_child_state_choice[ind_low],
        endog_grid_child_state_choice[ind_low],
        wealth_beginning_of_period,
        compute_utility,
        compute_marginal_utility,
        endog_grid_child_state_choice[1],
        value_child_state_choice[0],
        state_choice_vec,
        params,
    )

    return marg_utils, value_interp


def _interpolate_value_and_marg_util(
    policy_high: float,
    value_high: float,
    wealth_high: float,
    policy_low: float,
    value_low: float,
    wealth_low: float,
    new_wealth: float,
    compute_utility: Callable,
    compute_marginal_utility: Callable,
    endog_grid_min: float,
    value_min: float,
    state_choice_vec: jnp.ndarray,
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
        new_wealth (float): New endogenous wealth grid value.
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
    policy_interp, value_interp = interpolate_policy_and_check_value(
        policy_high=policy_high,
        value_high=value_high,
        wealth_high=wealth_high,
        policy_low=policy_low,
        value_low=value_low,
        wealth_low=wealth_low,
        new_wealth=new_wealth,
        compute_utility=compute_utility,
        endog_grid_min=endog_grid_min,
        value_min=value_min,
        state_choice_vec=state_choice_vec,
        params=params,
    )

    marg_utility_interp = compute_marginal_utility(
        consumption=policy_interp, params=params, **state_choice_vec
    )

    return marg_utility_interp, value_interp


def interpolate_policy_and_check_value(
    policy_high: float,
    value_high: float,
    wealth_high: float,
    policy_low: float,
    value_low: float,
    wealth_low: float,
    new_wealth: float,
    compute_utility: Callable,
    endog_grid_min: float,
    value_min: float,
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
    policy_interp, value_interp_on_grid = interpolate_policy_and_value(
        policy_high=policy_high,
        value_high=value_high,
        wealth_high=wealth_high,
        policy_low=policy_low,
        value_low=value_low,
        wealth_low=wealth_low,
        wealth_new=new_wealth,
    )

    utility = compute_utility(
        consumption=new_wealth,
        params=params,
        **state_choice_vec,
    )
    value_interp_closed_form = utility + params["beta"] * value_min

    credit_constraint = new_wealth < endog_grid_min
    value_interp = (
        credit_constraint * value_interp_closed_form
        + (1 - credit_constraint) * value_interp_on_grid
    )
    return policy_interp, value_interp
