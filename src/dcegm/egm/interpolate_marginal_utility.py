from typing import Callable, Dict, Tuple

from jax import numpy as jnp
from jax import vmap

from dcegm.interpolation.interp1d import interp_value_and_policy_on_wealth


def interpolate_value_and_marg_utility_on_next_period_wealth(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    wealth_beginning_of_period: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy in the child states and compute the marginal value.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
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

    interp_for_state_choice = vmap(
        interpolate_value_and_marg_utility_for_single_state_choice,
        in_axes=(None, None, 0, 0, 0, 0, 0, None),
    )
    return interp_for_state_choice(
        compute_marginal_utility,
        compute_utility,
        state_choice_vec,
        wealth_beginning_of_period,
        endog_grid_child_state_choice,
        policy_child_state_choice,
        value_child_state_choice,
        params,
    )


def interpolate_value_and_marg_utility_for_single_state_choice(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    wealth_beginning_of_period: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy in the child states and compute the marginal value.

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

    # Generate interpolation function for a single wealth point where the endogenous grid,
    # policy and value are fixed.
    def interp_on_single_wealth(wealth):
        policy_interp, value_interp = interp_value_and_policy_on_wealth(
            wealth=wealth,
            endog_grid=endog_grid_child_state_choice,
            policy=policy_child_state_choice,
            value=value_child_state_choice,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
        )
        marg_utility_interp = compute_marginal_utility(
            consumption=policy_interp, params=params, **state_choice_vec
        )
        return marg_utility_interp, value_interp

    # Vectorize interp_on_single_wealth over savings and income shock dimension
    vector_interp_func = vmap(vmap(interp_on_single_wealth))

    marg_utils, value_interp = vector_interp_func(wealth_beginning_of_period)

    return marg_utils, value_interp
