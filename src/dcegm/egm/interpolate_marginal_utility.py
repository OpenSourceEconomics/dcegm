from typing import Any, Callable, Dict, Tuple

from jax import numpy as jnp
from jax import vmap

from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)


def interpolate_value_and_marg_util(
    model_funcs,
    state_choice_vec: Dict[str, int],
    continuous_grids_info: Dict[str, Any],
    cont_grids_next_period: Dict[str, jnp.ndarray],
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    child_state_idxs: jnp.ndarray,
    params: Dict[str, float],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy for all child states and compute marginal utility.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility of consumption.
        compute_utility (callable): Function for calculating the utility of consumption.
        state_choice_vec (dict): Dictionary containing the state and choice of the agent.
        assets_beginning_of_next_period (jnp.ndarray): 2d array of shape
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
        has_second_continuous_state (bool): Boolean indicating whether the model
            features a second continuous state variable. If False, the only
            continuous state variable is consumption/savings.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - value_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated value function.
        - marg_util_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated marginal utilities for each wealth level and
            income shock.

    """
    wealth_child_states = cont_grids_next_period["assets_begin_of_period"][
        child_state_idxs
    ]
    compute_marginal_utility = model_funcs["compute_marginal_utility"]
    compute_utility = model_funcs["compute_utility"]
    discount_factor = model_funcs["read_funcs"]["discount_factor"](params)

    if continuous_grids_info["second_continuous_exists"]:
        continuous_state_child_states = cont_grids_next_period["second_continuous"][
            child_state_idxs
        ]
        regular_grid = continuous_grids_info["second_continuous_grid"]

        interp_for_single_state_choice = vmap(
            interp2d_value_and_marg_util_for_state_choice,
            in_axes=(
                None,
                None,
                0,
                None,
                0,
                0,
                0,
                0,
                0,
                None,
                None,
            ),  # discrete state-choice
        )

        return interp_for_single_state_choice(
            compute_marginal_utility,
            compute_utility,
            state_choice_vec,
            regular_grid,
            wealth_child_states,
            continuous_state_child_states,
            endog_grid_child_state_choice,
            policy_child_state_choice,
            value_child_state_choice,
            params,
            discount_factor,
        )

    else:
        interp_for_single_state_choice = vmap(
            interp1d_value_and_marg_util_for_state_choice,
            in_axes=(None, None, 0, 0, 0, 0, 0, None, None),  # discrete state-choice
        )

        return interp_for_single_state_choice(
            compute_marginal_utility,
            compute_utility,
            state_choice_vec,
            wealth_child_states,
            endog_grid_child_state_choice,
            policy_child_state_choice,
            value_child_state_choice,
            params,
            discount_factor,
        )


def interp1d_value_and_marg_util_for_state_choice(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    assets_beginning_of_next_period: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
    discount_factor: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy for given child state and compute marginal utility.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility of consumption.
        compute_utility (callable): Function for calculating the utility of consumption.
        state_choice_vec (dict): Dictionary containing the state and choice of the agent.
        assets_beginning_of_next_period (jnp.ndarray): 2d array of shape
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
        has_second_continuous_state (bool): Boolean indicating whether the model
            features a second continuous state variable. If False, the only
            continuous state variable is consumption/savings.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - marg_utils (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated marginal utilities for each wealth level and
            income shock.
        - value_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated value function.

    """

    def interp_on_single_wealth_point(wealth_point):
        policy_interp, value_interp = interp1d_policy_and_value_on_wealth(
            wealth=wealth_point,
            endog_grid=endog_grid_child_state_choice,
            policy=policy_child_state_choice,
            value=value_child_state_choice,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
            discount_factor=discount_factor,
        )
        marg_util_interp = compute_marginal_utility(
            consumption=policy_interp, params=params, **state_choice_vec
        )

        return value_interp, marg_util_interp

    interp_over_single_wealth_and_income_shock_draw = vmap(
        vmap(interp_on_single_wealth_point)  # income shocks
    )  # wealth grid

    value_interp, marg_util_interp = interp_over_single_wealth_and_income_shock_draw(
        assets_beginning_of_next_period
    )

    return value_interp, marg_util_interp


def interp2d_value_and_marg_util_for_state_choice(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    regular_grid: jnp.ndarray,
    assets_beginning_of_next_period: jnp.ndarray,
    continuous_state_beginning_of_next_period: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
    discount_factor: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy for given child state and compute marginal utility.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility of consumption.
        compute_utility (callable): Function for calculating the utility of consumption.
        state_choice_vec (dict): Dictionary containing the state and choice of the agent.
        assets_beginning_of_next_period (jnp.ndarray): 2d array of shape
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
        has_second_continuous_state (bool): Boolean indicating whether the model
            features a second continuous state variable. If False, the only
            continuous state variable is consumption/savings.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - marg_utils (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated marginal utilities for each wealth level and
            income shock.
        - value_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated value function.

    """

    def interp_on_single_wealth_point(wealth_point, second_cont_grid_point):

        policy_interp, value_interp = (
            interp2d_policy_and_value_on_wealth_and_regular_grid(
                regular_grid=regular_grid,
                wealth_grid=endog_grid_child_state_choice,
                policy_grid=policy_child_state_choice,
                value_grid=value_child_state_choice,
                wealth_point_to_interp=wealth_point,
                regular_point_to_interp=second_cont_grid_point,
                compute_utility=compute_utility,
                state_choice_vec=state_choice_vec,
                params=params,
                discount_factor=discount_factor,
            )
        )
        marg_util_interp = compute_marginal_utility(
            consumption=policy_interp,
            continuous_state=second_cont_grid_point,
            params=params,
            **state_choice_vec,
        )

        return value_interp, marg_util_interp

    # Outer vmap applies first
    interp_over_single_wealth_and_income_shock_draw = vmap(
        vmap(
            vmap(
                interp_on_single_wealth_point,
                in_axes=(0, None),  # income shocks
            ),
            in_axes=(0, None),  # wealth grid
        ),
        in_axes=(0, 0),  # continuous state grid
    )
    # Old points: regular grid and endog grid
    # New points: continuous state next period and wealth next period
    value_interp, marg_util_interp = interp_over_single_wealth_and_income_shock_draw(
        assets_beginning_of_next_period, continuous_state_beginning_of_next_period
    )

    return value_interp, marg_util_interp
