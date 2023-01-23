"""Auxiliary functions for the EGM algorithm."""
from typing import Callable
from typing import Tuple

import numpy as np


def compute_optimal_policy_and_value(
    marginal_utilities_exog_process,
    maximum_values_exog_process,
    discount_factor,
    interest_rate: float,
    choice: int,
    trans_vec_state: np.ndarray,
    exogenous_savings_grid: np.ndarray,
    compute_inverse_marginal_utility: Callable,
    compute_value: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal choice- and child-state specific policy and value function.

    Given the marginal utilities of possible child states and next period wealth we
    compute the optimal policy and current value functions by solving the euler equation
    and using the optimal consumption level in the bellman equation.

    Args:
        marginal_utilities_exog_process (np.ndarray): 2d array of shape
            (n_exog_processes, n_grid_wealth).
        maximum values_exog_process (np.ndarray): 2d array of shape
            (n_exog_processes, n_grid_wealth).
        discount_factor (float): The discount factor.
        interest_rate (float): The interest rate of capital.
        choice (int): The current discrete choice.
        trans_vec_state (np.ndarray): A vector containing for each possible exogenous
            process state the corresponding probability.
            Shape is (n_exog_processes).
        exogenous_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the exogenous savings grid.
        compute_inverse_marginal_utility (Callable): Function for calculating the
            inverse marginal utility, which takes the marginal utility as only input.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.

    Returns:
        tuple:

        - (np.ndarray): 2d array of the agent's period- and choice-specific
            consumption policy. Shape (2, n_grid_wealth + 1).
            Position [0,:] contains the endogenous grid over wealth M, and [1, :]
            stores the corresponding value of the policy function c(M, d).
        - (np.ndarray): 2d array of the agent's period- and choice-specific
            value function. Shape (2, n_grid_wealth + 1).  Position [0, :]
            contains the endogenous grid over wealth M, and [1, :] stores the
            corresponding value of the value function v(M, d).

    """
    current_policy, expected_value = solve_euler_equation(
        trans_vec_state,
        discount_factor,
        interest_rate,
        compute_inverse_marginal_utility,
        marginal_utilities_exog_process,
        maximum_values_exog_process,
    )

    current_policy_arr, current_value_arr = _create_current_policy_and_value_array(
        current_policy,
        expected_value,
        choice,
        exogenous_savings_grid,
        compute_value,
    )

    return current_policy_arr, current_value_arr


def solve_euler_equation(
    trans_vec_state,
    discount_factor,
    interest_rate,
    compute_inverse_marginal_utility,
    marginal_utilities,
    maximum_values,
):
    """Solve the Euler equation for given discrete choice and child states.

    We integrate over the exogenous process and income uncertainty and
    then apply the inverese marginal utility function.

    Args:
        trans_vec_state (np.ndarray): A vector containing for each possible exogenous
            process state the corresponding probability.
            Shape is (n_exog_processes).
        discount_factor (float): The discount factor.
        interest_rate (float): The interest rate of capital.
        compute_inverse_marginal_utility (callable): Function for calculating the
            inverse marginal utility, which takes the marginal utility as only input.
        marginal_utilities (np.ndarray): 2d array of shape
            (n_exog_processes, n_grid_wealth) with marginal utilities.
        maximum_values (np.ndarray): 2d array of shape
            (n_exog_processes, n_grid_wealth) with the maximum values.

    Returns:
        tuple:

        - (np.ndarray): 1d array of the agent's current choice-specific
            consumption policy. Has shape (n_grid_wealth,).
        - (np.ndarray): 1d array of the agent's current choice-specific
            expected value. Has shape (n_grid_wealth,).

    """
    # Integrate out uncertainty over exogenous process
    marginal_utility = trans_vec_state @ marginal_utilities
    expected_value = trans_vec_state @ maximum_values

    # RHS of Euler Eq., p. 337 IJRS (2017) by multiplying with marginal wealth
    rhs_euler = marginal_utility * (1 + interest_rate) * discount_factor
    current_policy = compute_inverse_marginal_utility(rhs_euler)

    return current_policy, expected_value


def _create_current_policy_and_value_array(
    current_policy: np.ndarray,
    expected_value: np.ndarray,
    current_choice: float,
    exogenous_savings_grid: np.ndarray,
    compute_value,
) -> Tuple[np.ndarray, np.ndarray]:
    """Store the current period policy and value functions.

    Args:
        current_policy (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the agent's current period policy rule.
        expected_value (np.ndarray): (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the agent's expected value of the next period.
        current_choice (int): The current discrete choice.
        exogenous_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the exogenous savings grid.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.

    Returns:
        tuple:

        - (np.ndarray): 2d array of the agent's period- and choice-specific
            consumption policy. Shape (2, n_grid_wealth + 1).
            Position [0,:] contains the endogenous grid over wealth M, and [1, :]
            stores the corresponding value of the policy function c(M, d).
        - (np.ndarray): 2d array of the agent's period- and choice-specific
            value function. Shape (2, n_grid_wealth + 1).
            Position [0, :] contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the value function v(M, d).

    """
    n_grid_wealth = exogenous_savings_grid.shape[0]

    endogenous_wealth_grid = exogenous_savings_grid + current_policy

    current_policy_container = np.zeros((2, n_grid_wealth + 1))
    current_policy_container[0, 1:] = endogenous_wealth_grid
    current_policy_container[1, 1:] = current_policy

    current_value_container = np.zeros((2, n_grid_wealth + 1))
    current_value_container[0, 1:] = endogenous_wealth_grid
    current_value_container[1, 0] = expected_value[0]
    current_value_container[1, 1:] = compute_value(
        current_policy, expected_value, current_choice
    )

    return current_policy_container, current_value_container
