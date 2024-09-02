from typing import Callable, Dict, Tuple

from jax import numpy as jnp
from jax import vmap

from dcegm.interpolation.interp1d import interpolate_policy_and_value_on_wealth


def interpolate_value_and_marg_util(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    wealth_beginning_of_period: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy for all child states and compute marginal utility.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility of consumption.
        compute_utility (callable): Function for calculating the utility of consumption.
        state_choice_vec (dict): Dictionary containing the state and choice of the agent.
        wealth_beginning_of_period (jnp.ndarray): 2d array of shape
            (n_quad_stochastic, n_grid_wealth,) containing the agent's beginning of
            period wealth.
        endog_grid_child_state_choice (jnp.ndarray): 1d array containing the endogenous
            wealth grid of the child state/choice pair. Shape (n_grid_wealth,).
        policy_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding policy function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        value_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding value function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - value_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated value function.
        - marg_util_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated marginal utilities for each wealth level and
            income shock.

    """

    interp_for_single_state_choice = vmap(
        interpolate_value_and_marg_util_for_single_state_choice,
        in_axes=(None, None, 0, 0, 0, 0, 0, None),
    )

    return interp_for_single_state_choice(
        compute_marginal_utility,
        compute_utility,
        state_choice_vec,
        wealth_beginning_of_period,
        endog_grid_child_state_choice,
        policy_child_state_choice,
        value_child_state_choice,
        params,
    )


def interpolate_value_and_marg_util_for_single_state_choice(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    wealth_beginning_of_period: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy for given child state and compute marginal utility.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility of consumption.
        compute_utility (callable): Function for calculating the utility of consumption.
        state_choice_vec (dict): Dictionary containing the state and choice of the agent.
        wealth_beginning_of_period (jnp.ndarray): 2d array of shape
            (n_quad_stochastic, n_grid_wealth,) containing the agent's beginning of
            period wealth.
        endog_grid_child_state_choice (jnp.ndarray): 1d array containing the endogenous
            wealth grid of the child state/choice pair. Shape (n_grid_wealth,).
        policy_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding policy function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        value_child_state_choice (jnp.ndarray): 1d array containing the
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

    def interp_on_single_wealth_point(wealth):
        policy_interp, value_interp = interpolate_policy_and_value_on_wealth(
            wealth=wealth,
            endog_grid=endog_grid_child_state_choice,
            policy=policy_child_state_choice,
            value=value_child_state_choice,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
        )
        marg_util_interp = compute_marginal_utility(
            consumption=policy_interp, params=params, **state_choice_vec
        )
        return value_interp, marg_util_interp

    # Vectorize over savings and income shock dimension
    interp_for_savings_point_and_income_shock_draw = vmap(
        vmap(interp_on_single_wealth_point)
    )
    value_interp, marg_util_interp = interp_for_savings_point_and_income_shock_draw(
        wealth_beginning_of_period
    )

    return value_interp, marg_util_interp
