"""Auxiliary functions for the EGM algorithm."""

from typing import Callable, Dict, Tuple

import numpy as np
from jax import numpy as jnp
from jax import vmap


def calculate_candidate_solutions_from_euler_equation(
    continuous_grids_info: np.ndarray,
    marg_util_next: jnp.ndarray,
    emax_next: jnp.ndarray,
    state_choice_mat: np.ndarray,
    idx_post_decision_child_states: np.ndarray,
    model_funcs: Dict[str, Callable],
    has_second_continuous_state: bool,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate candidates for the optimal policy and value function."""

    feasible_marg_utils_child = jnp.take(
        marg_util_next, idx_post_decision_child_states, axis=0
    )
    feasible_emax_child = jnp.take(emax_next, idx_post_decision_child_states, axis=0)

    if has_second_continuous_state:
        (
            endog_grid,
            policy,
            value,
            expected_value,
        ) = vmap(
            vmap(
                vmap(
                    compute_optimal_policy_and_value_wrapper,
                    in_axes=(1, 1, None, 0, None, None, None),  # assets
                ),
                in_axes=(1, 1, 0, None, None, None, None),  # second continuous state
            ),
            in_axes=(0, 0, None, None, 0, None, None),  # discrete states choices
        )(
            feasible_marg_utils_child,
            feasible_emax_child,
            continuous_grids_info["second_continuous_grid"],
            continuous_grids_info["assets_grid_end_of_period"],
            state_choice_mat,
            model_funcs,
            params,
        )
    else:
        (
            endog_grid,
            policy,
            value,
            expected_value,
        ) = vmap(
            vmap(
                compute_optimal_policy_and_value,
                in_axes=(1, 1, 0, None, None, None),  # assets grid
            ),
            in_axes=(0, 0, None, 0, None, None),  # states and choices
        )(
            feasible_marg_utils_child,
            feasible_emax_child,
            continuous_grids_info["assets_grid_end_of_period"],
            state_choice_mat,
            model_funcs,
            params,
        )

    return (
        endog_grid,
        value,
        policy,
        expected_value,
    )


def compute_optimal_policy_and_value_wrapper(
    marg_util_next: np.ndarray,
    emax_next: np.ndarray,
    second_continuous_grid: np.ndarray,
    assets_grid_end_of_period: np.ndarray,
    state_choice_vec: Dict,
    model_funcs: Dict[str, Callable],
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Write second continuous grid point into state_choice_vec."""
    state_choice_vec["continuous_state"] = second_continuous_grid

    return compute_optimal_policy_and_value(
        marg_util_next,
        emax_next,
        assets_grid_end_of_period,
        state_choice_vec,
        model_funcs,
        params,
    )


def compute_optimal_policy_and_value(
    marg_util_next: np.ndarray,
    emax_next: np.ndarray,
    assets_grid_end_of_period: np.ndarray,
    state_choice_vec: Dict,
    model_funcs: Dict[str, Callable],
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute optimal child-state- and choice-specific policy and value function.

    Given the marginal utilities of possible child states and next period wealth, we
    compute the optimal policy and value functions by solving the euler equation
    and using the optimal consumption level in the bellman equation.

    Args:
        marg_utils (np.ndarray): 1d array of shape (n_stochastic_states,) containing
            the state-choice specific marginal utilities for a given point on
            the savings grid.
        emax (np.ndarray): 1d array of shape (n_stochastic_states,) containing
            the state-choice specific expected maximum value for a given point on
            the savings grid.
        assets_grid_end_of_period (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the exogenous savings grid.
        trans_vec_state (np.ndarray): 1d array of shape (n_stochastic_states,) containing
            for each exogenous process state the corresponding transition probability.
        state_choice_vec (np.ndarray): A dictionary containing the states and a
        corresponding admissible choice of a particular state choice vector.
        compute_inverse_marginal_utility (Callable): Function for calculating the
            inverse marginal utility, which takes the marginal utility as only input.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
        params (dict): Dictionary of model parameters.

    Returns:
        tuple:

        - endog_grid (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific endogenous grid.
        - policy (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific policy function.
        - value (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific value function.
        - expected_value_zero_savings (float): The agent's expected value given that
            she saves nothing.

    """
    compute_inverse_marginal_utility = model_funcs["compute_inverse_marginal_utility"]
    compute_utility = model_funcs["compute_utility"]
    compute_stochastic_transition_vec = model_funcs["compute_stochastic_transition_vec"]

    discount_factor = model_funcs["read_funcs"]["discount_factor"](params)
    interest_rate = model_funcs["read_funcs"]["interest_rate"](params)

    policy, expected_value = solve_euler_equation(
        state_choice_vec=state_choice_vec,
        marg_util_next=marg_util_next,
        emax_next=emax_next,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
        compute_stochastic_transition_vec=compute_stochastic_transition_vec,
        params=params,
        discount_factor=discount_factor,
        interest_rate=interest_rate,
    )
    endog_grid = assets_grid_end_of_period + policy

    utility = compute_utility(consumption=policy, params=params, **state_choice_vec)
    value = utility + discount_factor * expected_value

    return endog_grid, policy, value, expected_value


def solve_euler_equation(
    state_choice_vec: dict,
    marg_util_next: np.ndarray,
    emax_next: np.ndarray,
    compute_inverse_marginal_utility: Callable,
    compute_stochastic_transition_vec: Callable,
    params: Dict[str, float],
    discount_factor: float,
    interest_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Euler equation for given discrete choice and child states.

    We integrate over the exogenous process and income uncertainty and
    then apply the inverese marginal utility function.

    Args:
        marg_utils (np.ndarray): 1d array of shape (n_stochastic_states,) containing
            the state-choice specific marginal utilities for a given point on
            the savings grid.
        emax (np.ndarray): 1d array of shape (n_stochastic_states,) containing
            the state-choice specific expected maximum value for a given point on
            the savings grid.
        trans_vec_state (np.ndarray): 1d array of shape (n_stochastic_states,) containing
            for each exogenous process state the corresponding transition probability.
        compute_inverse_marginal_utility (callable): Function for calculating the
            inverse marginal utility, which takes the marginal utility as only input.
            (n_stochastic_states, n_grid_wealth) with the maximum values.
        params (dict): Dictionary of model parameters.

    Returns:
        tuple:

        - policy (np.ndarray): 1d array of the agent's current state- and
            choice-specific consumption policy. Has shape (n_grid_wealth,).
        - expected_value (np.ndarray): 1d array of the agent's current state- and
            choice-specific expected value. Has shape (n_grid_wealth,).

    """

    transition_vec = compute_stochastic_transition_vec(
        params=params, **state_choice_vec
    )

    # Integrate out uncertainty over exogenous processes
    marginal_utility_next = jnp.nansum(transition_vec * marg_util_next)
    expected_value = jnp.nansum(transition_vec * emax_next)

    # RHS of Euler Eq., p. 337 IJRS (2017) by multiplying with marginal wealth
    rhs_euler = marginal_utility_next * (1 + interest_rate) * discount_factor

    policy = compute_inverse_marginal_utility(
        marginal_utility=rhs_euler,
        params=params,
        **state_choice_vec,
    )

    return policy, expected_value
