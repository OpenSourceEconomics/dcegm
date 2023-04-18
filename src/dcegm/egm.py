"""Auxiliary functions for the EGM algorithm."""
from typing import Callable
from typing import Tuple

import numpy as np


def compute_optimal_policy_and_value(
    marginal_utilities_exog_process: np.ndarray,
    maximum_values_exog_process: np.ndarray,
    trans_vec_state: np.ndarray,
    exogenous_savings_grid: np.ndarray,
    discount_factor: float,
    interest_rate: float,
    choice: int,
    compute_inverse_marginal_utility: Callable,
    compute_value: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute optimal choice- and child-state specific policy and value function.

    Given the marginal utilities of possible child states and next period wealth, we
    compute the optimal policy and value functions by solving the euler equation
    and using the optimal consumption level in the bellman equation.

    Args:
        marginal_utilities_exog_process (np.ndarray): 2d array of shape
            (n_exog_processes, n_grid_wealth).
        maximum values_exog_process (np.ndarray): 2d array of shape
            (n_exog_processes, n_grid_wealth).
        trans_vec_state (np.ndarray): A vector containing for each possible exogenous
            process state the corresponding probability.
            Shape is (n_exog_processes).
        exogenous_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the exogenous savings grid.
        discount_factor (float): The discount factor.
        interest_rate (float): The interest rate of capital.
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

    """
    n_grid_wealth = len(exogenous_savings_grid)

    _policy, expected_value = solve_euler_equation(
        marginal_utilities=marginal_utilities_exog_process,
        maximum_values=maximum_values_exog_process,
        trans_vec_state=trans_vec_state,
        discount_factor=discount_factor,
        interest_rate=interest_rate,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
    )
    endog_grid = np.zeros(n_grid_wealth + 1)
    endog_grid[1:] = exogenous_savings_grid + _policy

    policy = np.zeros(n_grid_wealth + 1)
    policy[1:] = _policy

    value = np.zeros(n_grid_wealth + 1)
    value[0] = expected_value[0]
    value[1:] = compute_value(_policy, expected_value, choice)

    return endog_grid, policy, value


def solve_euler_equation(
    marginal_utilities: np.ndarray,
    maximum_values: np.ndarray,
    trans_vec_state: np.ndarray,
    discount_factor: float,
    interest_rate: float,
    compute_inverse_marginal_utility: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Euler equation for given discrete choice and child states.

    We integrate over the exogenous process and income uncertainty and
    then apply the inverese marginal utility function.

    Args:
        marginal_utilities (np.ndarray): 2d array of shape
            (n_exog_processes, n_grid_wealth) with marginal utilities.
        maximum_values (np.ndarray): 2d array of shape
        trans_vec_state (np.ndarray): 1d array of shape (n_exog_processes,)
            containing the state probabilities of each exogenous process.
        discount_factor (float): The discount factor.
        interest_rate (float): The interest rate of capital.
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
    marginal_utility = trans_vec_state @ marginal_utilities
    expected_value = trans_vec_state @ maximum_values

    # RHS of Euler Eq., p. 337 IJRS (2017) by multiplying with marginal wealth
    rhs_euler = marginal_utility * (1 + interest_rate) * discount_factor
    policy = compute_inverse_marginal_utility(rhs_euler)

    return policy, expected_value
