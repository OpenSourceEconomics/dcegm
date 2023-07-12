"""Auxiliary functions for the EGM algorithm."""
from typing import Callable
from typing import Tuple

import numpy as np
from jax import numpy as jnp
from jax import vmap


def calculate_candidate_solutions_from_euler_equation(
    marg_util: np.ndarray,
    emax: np.ndarray,
    idx_state_choices_period: np.ndarray,
    map_state_to_post_decision_child_nodes: Callable,
    exogenous_savings_grid: np.ndarray,
    transition_vector_by_state: Callable,
    discount_factor: float,
    interest_rate: float,
    state_choices_period: np.ndarray,
    compute_inverse_marginal_utility: Callable,
    compute_value: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate candidates for the optimal policy and value function."""
    feasible_marg_utils, feasible_emax = _get_post_decision_marg_utils_and_emax(
        marg_util_next=marg_util,
        emax_next=emax,
        idx_state_choice_combs=idx_state_choices_period,
        map_state_to_post_decision_child_nodes=map_state_to_post_decision_child_nodes,
    )

    (
        endog_grid_candidate,
        policy_candidate,
        value_candiadate,
        expected_values,
    ) = vmap(
        vmap(
            compute_optimal_policy_and_value,
            in_axes=(1, 1, 0, None, None, None, None, None, None),  # savings grid
        ),
        in_axes=(0, 0, None, None, None, None, 0, None, None),  # states and choices
    )(
        feasible_marg_utils,
        feasible_emax,
        exogenous_savings_grid,
        transition_vector_by_state,
        discount_factor,
        interest_rate,
        state_choices_period,
        compute_inverse_marginal_utility,
        compute_value,
    )
    return (
        endog_grid_candidate,
        value_candiadate,
        policy_candidate,
        expected_values,
    )


def compute_optimal_policy_and_value(
    marg_utils: np.ndarray,
    emax: np.ndarray,
    exogenous_savings_grid: np.ndarray,
    transition_vector_by_state: Callable,
    discount_factor: float,
    interest_rate: float,
    state_choice_subspace: np.ndarray,
    compute_inverse_marginal_utility: Callable,
    compute_value: Callable,
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
        discount_factor (float): The discount factor.
        interest_rate (float): The interest rate on capital.
        choice (int): The current discrete choice.
        compute_inverse_marginal_utility (Callable): Function for calculating the
            inverse marginal utility, which takes the marginal utility as only input.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.

    Returns:
        tuple:

        - endog_grid (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific endogenous grid.
        - policy (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific policy function.
        - value (np.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific value function.
        expected_value_zero_savings (float): The agent's expected value given that she
            saves zero.

    """
    state_vec = state_choice_subspace[:-1]
    choice = state_choice_subspace[-1]
    transition_probs = transition_vector_by_state(state_vec)

    policy, expected_value = solve_euler_equation(
        marg_utils=marg_utils,
        emax=emax,
        transition_probs=transition_probs,
        discount_factor=discount_factor,
        interest_rate=interest_rate,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
    )
    endog_grid = exogenous_savings_grid + policy

    value = compute_value(policy, expected_value, choice)

    return endog_grid, policy, value, expected_value


def solve_euler_equation(
    marg_utils: np.ndarray,
    emax: np.ndarray,
    transition_probs: np.ndarray,
    discount_factor: float,
    interest_rate: float,
    compute_inverse_marginal_utility: Callable,
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
        discount_factor (float): The discount factor.
        interest_rate (float): The interest rate on capital.
        compute_inverse_marginal_utility (callable): Function for calculating the
            inverse marginal utility, which takes the marginal utility as only input.
            (n_exog_processes, n_grid_wealth) with the maximum values.

    Returns:
        tuple:

        - policy (np.ndarray): 1d array of the agent's current state- and
            choice-specific consumption policy. Has shape (n_grid_wealth,).
        - expected_value (np.ndarray): 1d array of the agent's current state- and
            choice-specific expected value. Has shape (n_grid_wealth,).

    """
    # Integrate out uncertainty over exogenous process
    marginal_utility = transition_probs @ marg_utils
    expected_value = transition_probs @ emax

    # RHS of Euler Eq., p. 337 IJRS (2017) by multiplying with marginal wealth
    rhs_euler = marginal_utility * (1 + interest_rate) * discount_factor
    policy = compute_inverse_marginal_utility(rhs_euler)

    return policy, expected_value


def _get_post_decision_marg_utils_and_emax(
    marg_util_next,
    emax_next,
    idx_state_choice_combs,
    map_state_to_post_decision_child_nodes,
):
    """Get marginal utility and expected maximum value of post-decision child states.

    Args:
        marg_util_next (np.ndarray): 2d array of shape (n_choices, n_grid_wealth)
            containing the choice-specific marginal utilities of the next period,
            i.e. t + 1.
        emax_next (np.ndarray): 2d array of shape (n_choices, n_grid_wealth)
            containing the choice-specific expected maximum values of the next period,
            i.e. t + 1.
        idx_state_choice_combs (np.ndarray): Indexer for the state choice combinations
            that are feasible in the current period.
        map_state_to_post_decision_child_nodes (np.ndarray): Indexer for the child nodes
            that can be reached from the current state.

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
    idx_post_decision_child_states = map_state_to_post_decision_child_nodes[
        idx_state_choice_combs
    ]

    # state-choice specific
    marg_utils_child = jnp.take(marg_util_next, idx_post_decision_child_states, axis=0)
    emax_child = jnp.take(emax_next, idx_post_decision_child_states, axis=0)

    return marg_utils_child, emax_child
