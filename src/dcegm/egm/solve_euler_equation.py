"""Auxiliary functions for the EGM algorithm."""

from typing import Callable, Dict, Tuple

import numpy as np
from jax import numpy as jnp
from jax import vmap


def calculate_candidate_solutions_from_euler_equation(
    exogenous_savings_grid: np.ndarray,
    marg_util: jnp.ndarray,
    emax: jnp.ndarray,
    state_choice_vec: np.ndarray,
    idx_post_decision_child_states: np.ndarray,
    compute_utility: Callable,
    compute_inverse_marginal_utility: Callable,
    compute_exog_transition_vec: Callable,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate candidates for the optimal policy and value function."""
    feasible_marg_utils, feasible_emax = _get_post_decision_marg_utils_and_emax(
        marg_util_next=marg_util,
        emax_next=emax,
        idx_post_decision_child_states=idx_post_decision_child_states,
    )

    # transform exog_transition_mat to matrix with same shape as state_choice_vec

    (
        endog_grid,
        policy,
        value,
        expected_value,
    ) = vmap(
        vmap(
            compute_optimal_policy_and_value,
            in_axes=(1, 1, 0, None, None, None, None, None),  # savings grid
        ),
        in_axes=(0, 0, None, 0, None, None, None, None),  # states and choices
    )(
        feasible_marg_utils,
        feasible_emax,
        exogenous_savings_grid,
        state_choice_vec,
        compute_inverse_marginal_utility,
        compute_utility,
        compute_exog_transition_vec,
        params,
    )
    return (
        endog_grid,
        value,
        policy,
        expected_value,
    )


def compute_optimal_policy_and_value(
    marg_utils: np.ndarray,
    emax: np.ndarray,
    exogenous_savings_grid: np.ndarray,
    state_choice_vec: Dict,
    compute_inverse_marginal_utility: Callable,
    compute_utility: Callable,
    compute_exog_transition_vec: Callable,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute optimal child-state- and choice-specific policy and value function.

    Given the marginal utilities of possible child states and next period wealth, we
    compute the optimal policy and value functions by solving the euler equation
    and using the optimal consumption level in the bellman equation.

    Args:
        marg_utils (np.ndarray): 1d array of shape (n_exog_processes,) containing
            the state-choice specific marginal utilities for a given point on
            the savings grid.
        emax (np.ndarray): 1d array of shape (n_exog_processes,) containing
            the state-choice specific expected maximum value for a given point on
            the savings grid.
        exogenous_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the exogenous savings grid.
        trans_vec_state (np.ndarray): 1d array of shape (n_exog_processes,) containing
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

    policy, expected_value = solve_euler_equation(
        state_choice_vec=state_choice_vec,
        marg_utils=marg_utils,
        emax=emax,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
        compute_exog_transition_vec=compute_exog_transition_vec,
        params=params,
    )
    endog_grid = exogenous_savings_grid + policy

    utility = compute_utility(consumption=policy, params=params, **state_choice_vec)
    value = utility + params["beta"] * expected_value

    return endog_grid, policy, value, expected_value


def solve_euler_equation(
    state_choice_vec: dict,
    marg_utils: np.ndarray,
    emax: np.ndarray,
    compute_inverse_marginal_utility: Callable,
    compute_exog_transition_vec: Callable,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Euler equation for given discrete choice and child states.

    We integrate over the exogenous process and income uncertainty and
    then apply the inverese marginal utility function.

    Args:
        marg_utils (np.ndarray): 1d array of shape (n_exog_processes,) containing
            the state-choice specific marginal utilities for a given point on
            the savings grid.
        emax (np.ndarray): 1d array of shape (n_exog_processes,) containing
            the state-choice specific expected maximum value for a given point on
            the savings grid.
        trans_vec_state (np.ndarray): 1d array of shape (n_exog_processes,) containing
            for each exogenous process state the corresponding transition probability.
        compute_inverse_marginal_utility (callable): Function for calculating the
            inverse marginal utility, which takes the marginal utility as only input.
            (n_exog_processes, n_grid_wealth) with the maximum values.
        params (dict): Dictionary of model parameters.

    Returns:
        tuple:

        - policy (np.ndarray): 1d array of the agent's current state- and
            choice-specific consumption policy. Has shape (n_grid_wealth,).
        - expected_value (np.ndarray): 1d array of the agent's current state- and
            choice-specific expected value. Has shape (n_grid_wealth,).

    """

    transition_vec = compute_exog_transition_vec(params=params, **state_choice_vec)

    # Integrate out uncertainty over exogenous processes
    marginal_utility = jnp.nansum(transition_vec * marg_utils)
    expected_value = jnp.nansum(transition_vec * emax)

    # RHS of Euler Eq., p. 337 IJRS (2017) by multiplying with marginal wealth
    rhs_euler = marginal_utility * (1 + params["interest_rate"]) * params["beta"]

    policy = compute_inverse_marginal_utility(
        marginal_utility=rhs_euler,
        params=params,
        **state_choice_vec,
    )

    return policy, expected_value


def _get_post_decision_marg_utils_and_emax(
    marg_util_next,
    emax_next,
    idx_post_decision_child_states,
):
    """Get marginal utility and expected maximum value of post-decision child states.

    Args:
        marg_util_next (jnp.ndarray): 2d array of shape (n_choices, n_grid_wealth)
            containing the choice-specific marginal utilities of the next period,
            i.e. t + 1.
        emax_next (jnp.ndarray): 2d array of shape (n_choices, n_grid_wealth)
            containing the choice-specific expected maximum values of the next period,
            i.e. t + 1.
        idx_post_decision_child_states (jnp.ndarray): 2d array of shape
            (n_state_choice_combinations_period, n_exog_states) containing the
            indexes of the feasible post-decision child states in the current period t.
            The indexes are normalized such that they start at 0.

    Returns:
        tuple:

        - marg_utils_child (np.ndarray): 3d array of shape
            (n_child_states, n_exog_processes, n_grid_wealth) containing the
            state-choice specific marginal utilities of the child states in
            the current period t.
        - emax_child (np.ndarray): 3d array of shape
            (n_child_states, n_exog_processes, n_grid_wealth) containing the
            state-choice specific expected maximum values of the child states
            in the current period t.

    """

    # state-choice specific
    marg_utils_child = jnp.take(marg_util_next, idx_post_decision_child_states, axis=0)
    emax_child = jnp.take(emax_next, idx_post_decision_child_states, axis=0)

    return marg_utils_child, emax_child
