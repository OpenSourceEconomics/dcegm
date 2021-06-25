"""Implementation of the EGM algorithm."""
import copy
from typing import Callable, Dict, Tuple


import numpy as np
import pandas as pd

from scipy import interpolate


def call_egm_step(
    period: int,
    state: int,
    policy: np.ndarray,
    value: np.ndarray,
    savings_grid: np.ndarray,
    quad_points_normal: np.ndarray,
    quad_weights: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_func: Callable,
    marginal_utility_func: Callable,
    inv_marginal_utility_func: Callable,
    compute_value_function: Callable,
    compute_next_period_wealth_matrix: Callable,
    compute_next_period_marg_wealth_matrix: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calls the Endogenous Grid Method (EGM step).

    Args:
        period (int): Current period t.
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        policy (np.ndarray): Multi-dimensional array of choice-specific
            consumption policy. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        value (np.ndarray): Multi-dimensional array of choice-specific values of the
            the value function. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        savings_grid (np.ndarray): Array of shape n_wealth_grid denoting the
            exogenous savings grid.
        quad_points_normal (np.ndarray): Array of shape (n_quad_stochastic,)
            containing (normally distributed) stochastic components.
        quad_weights (np.ndarray): Weights associated with the quadrature points.
            Will be used for integration over the stochastic income component
            in the Euler equation below. Also of shape (n_quad_stochastic,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_func (callable): The agent's utility function.
        marginal_utility_func (callable): Marginal utility function.
        inv_marginal_utility_func (callable): Inverse of the marginal utility
            function.
        compute_value_function (callable): Function to compute the agent's next-period
            value function, which is an array of shape
            (n_choices, n_quad_stochastic * n_grid_wealth).
        compute_next_period_wealth_matrix (callable): Function to compute next
            period wealth matrix which is an array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        compute_next_period_marg_wealth_matrix (callable): Function to compute next
            period wealth matrix which is an array of all possible next period
            marginal wealths with shape (n_quad_stochastic, n_grid_wealth).

    Returns:
        (tuple): Tuple containing:

        - policy (np.ndarray): Array storing levels of optimal consumption policy.
        - value (np.ndarray): Array Value function.
    """
    # 1) Policy: Current period consumption
    matrix_next_period_wealth = compute_next_period_wealth_matrix(
        period, state, savings_grid, quad_points_normal, params, options
    )
    matrix_marginal_wealth = compute_next_period_marg_wealth_matrix(params, options)

    next_period_consumption = get_next_period_consumption(
        period, policy, matrix_next_period_wealth
    )
    next_period_marginal_utility = marginal_utility_func(
        next_period_consumption, params
    )

    # RHS of Euler Eq., p. 337 IJRS (2017)
    # Integrate out uncertainty over stochastic income y
    rhs_euler = get_rhs_euler(
        next_period_marginal_utility,
        matrix_next_period_wealth,
        matrix_marginal_wealth,
        quad_weights,
    )
    current_period_consumption = get_current_period_consumption(
        rhs_euler, params, inv_marginal_utility_func
    )

    # 2) Value function: Current period value
    next_period_value = get_next_period_value(
        period,
        value,
        matrix_next_period_wealth,
        params,
        options,
        utility_func,
        compute_value_function,
    )
    current_period_utility = utility_func(current_period_consumption, params)
    expected_value = get_expected_value(
        next_period_value, matrix_next_period_wealth, quad_weights
    )
    current_period_value = get_current_period_value(
        current_period_utility, expected_value, state, params
    )

    # 3) Endogenous wealth grid
    endog_wealth_grid = savings_grid + current_period_consumption

    # 4) Update policy and consumption function
    # If no discrete choices to make, only one state, i.e. one column with index = 0
    state_index = 0 if options["n_discrete_choices"] < 2 else state

    policy[period, state_index, 0, 1:] = endog_wealth_grid
    policy[period, state_index, 1, 1:] = current_period_consumption

    value[period, state_index, 0, 1:] = endog_wealth_grid
    value[period, state_index, 1, 1:] = current_period_value
    value[period, state_index, 1, 0] = expected_value[0]

    return policy, value


def get_next_period_consumption(
    period: int,
    policy: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
) -> np.ndarray:
    """Computes consumption in the next period via linear interpolation.

    Extrapolate lineary in wealth regions to larger than max_wealth.

    Args:
        period (int): Current period t.
        policy (np.ndarray): Multi-dimensional array of choice-specific consumption
            policy. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).

    Returns:
        next_period_consumption_interp (np.ndarray): Array of next period
            consumption of shape (n_grid_wealth,).
    """
    next_period_wealth = policy[period + 1, 0, 0, :]
    next_period_consumption = policy[period + 1, 0, 1, :]

    interpolation_func = interpolate.interp1d(
        next_period_wealth,
        next_period_consumption,
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )
    next_period_consumption_interp = interpolation_func(
        matrix_next_period_wealth
    ).flatten("F")

    return next_period_consumption_interp


def get_current_period_consumption(
    rhs_euler: np.ndarray, params: pd.DataFrame, inv_marginal_utility_func: Callable
) -> np.ndarray:
    """Computes consumption in the current period.

    Args:
        rhs_euler (np.ndarray): Right-hand side of the Euler equation.
            Shape (n_grid_wealth,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        inv_marginal_utility_func (callable): Inverse of the marginal utility
            function.

    Returns:
        current_period_consumption (np.ndarray):
    """
    beta = params.loc[("beta", "beta"), "value"]
    current_period_consumption = inv_marginal_utility_func(beta * rhs_euler, params)

    return current_period_consumption


def get_rhs_euler(
    next_period_marginal_utility: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    matrix_marginal_wealth: np.ndarray,
    quad_weights: np.ndarray,
) -> np.ndarray:
    """Computes the right-hand side of the Euler equation, p. 337 IJRS (2017).

    Args:
        next_period_marginal_utility (np.ndarray): Array of next period's
            marginal utility of shape (n_grid_wealth,).
        matrix_next_period_wealth(np.ndarray): Array of all possible next
            period wealths. Shape (n_quad_stochastic, n_wealth_grid).
        matrix_marginal_wealth(np.ndarray): Array of marginal next period wealths.
            Shape (n_quad_stochastic, n_wealth_grid).
        quad_weights (np.ndarray): Weights associated with the quadrature points
            of shape (n_quad_stochastic,). Used for integration over the
            stochastic income component in the Euler equation.

    Returns:
        rhs_euler (np.ndarray): Right-hand side of the Euler equation.
            Shape (n_grid_wealth,).
    """
    next_period_marginal_utility = next_period_marginal_utility.reshape(
        matrix_next_period_wealth.shape, order="F"
    )

    rhs_euler = np.dot(
        quad_weights.T,
        np.multiply(next_period_marginal_utility, matrix_marginal_wealth),
    )

    return rhs_euler


def get_next_period_value(
    period: int,
    value: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_func: Callable,
    compute_value_function: Callable,
) -> np.ndarray:
    """Computes the next-period (choice-specific) value function.

    Args:
        period (int): Current period t.
        value (np.ndarray): Multi-dimensional array of choice-specific values of the
            the value function. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_func (callable): The agent's utility function.
        compute_value_function (callable): Function to compute the agent's value
            function, which is an array of shape
            (n_choices, n_quad_stochastic * n_grid_wealth).

    Returns:
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """
    delta = params.loc[("delta", "delta"), "value"]
    n_periods, n_choices = options["n_periods"], options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else range(n_choices)

    next_period_value = np.empty(
        (
            n_choices,
            matrix_next_period_wealth.shape[0] * matrix_next_period_wealth.shape[1],
        )
    )

    for index, state in enumerate(choice_range):
        if period + 1 == n_periods - 1:  # Final period
            next_period_value[index, :] = (
                utility_func(
                    matrix_next_period_wealth,
                    params,
                ).flatten("F")
                - delta * state
            )
        else:
            next_period_value[index, :] = compute_value_function(
                value,
                matrix_next_period_wealth,
                period + 1,
                state,
                utility_func,
                params,
                options,
            )

    return next_period_value


def get_current_period_value(
    current_period_utility: np.ndarray,
    expected_value: np.ndarray,
    state: int,
    params: pd.DataFrame,
) -> np.ndarray:
    """Computes value of the current period.

    Args:
        current_period_utility (np.ndarray): Array of current period utility
            of shape (n_grid_wealth,).
        expected_value (np.ndarray): Array of current period's expected value of
            next_period. Shape (n_grid_wealth,).
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].

    Returns:
        current_period_value (np.ndarray): Array of current period value
            function of shape n_grid_wealth.
    """
    delta = params.loc[("delta", "delta"), "value"]  # disutility of work
    beta = params.loc[("beta", "beta"), "value"]  # discount factor

    current_period_value = (
        current_period_utility - delta * state + beta * expected_value
    )

    return current_period_value


def get_expected_value(
    next_period_value: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    quad_weights: np.ndarray,
) -> np.ndarray:
    """Computes the expected value of the next period.

    Args:
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth)
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        quad_weights (np.ndarray): Weights associated with the stochastic
            quadrature points of shape (n_quad_stochastic,).

    Returns:
        expected_value (np.ndarray): Array of current period's expected value of
            next_period. Shape (n_grid_wealth,).
    """
    expected_value = np.dot(
        quad_weights.T,
        next_period_value[0, :].reshape(matrix_next_period_wealth.shape, order="F"),
    )

    return expected_value


def set_first_elements_to_zero(
    policy: np.ndarray,
    value: np.ndarray,
    options: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sets first elements of endogenous wealth grid and consumption policy to zero.

    Args:
        policy (np.ndarray): Multi-dimensional array of choice-specific
            consumption policy. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        value (np.ndarray): Multi-dimensional array of choice-specific values of the
            value function. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        expected_value (np.ndarray): Array of current period's expected value of
            next_period. Shape (n_grid_wealth,).
        options (dict): Options dictionary.

    Returns:
        (tuple): Tuple containing:
        - policy (np.ndarray): Multi-dimensional array of choice-specific
            consumption policy. The first elements in the endogenous wealth
            grid (last dimension) have been adjusted.
        - value (np.ndarray): Multi-dimensional array of choice-specific values of
            the value function. The first elements in the endogenous wealth
            grid (last dimension) have been adjusted.

        Both arrays have shape (n_periods, n_choices, 2, n_grid_wealth + 1).
    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]

    # Add point M_0 = 0 to the endogenous wealth grid in both the
    # policy and value function arrays
    for period in range(n_periods):
        for state in range(n_choices):
            policy[period, state, 0, 0] = 0
            value[period, state, 0, 0] = 0

            # Add corresponding consumption point c(M=0, d) = 0
            policy[period, state, 1, 0] = 0

    return policy, value


def solve_final_period(
    policy: np.ndarray,
    value: np.ndarray,
    savings_grid: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_func: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes solution to final period for consumption policy and value function.

    Args:
        policy (np.ndarray): Multi-dimensional array of choice-specific
            consumption policy. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        value (np.ndarray): Multi-dimensional array of choice-specific values of the
            the value function. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        savings_grid (np.ndarray): Array of shape (n_wealth_grid,) denoting the
            exogenous savings grid.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_func (callable): The agent's utility function.

    Returns:
        (tuple): Tuple containing:
        - policy (np.ndarray): Multi-dimensional array of choice-specific
            consumption policy with solution for final period.
        - value (np.ndarray): Multi-dimensional array of choice-specific values of
            the value function with solution for final period.

        Both arrays have shape (n_periods, n_choices, 2, n_grid_wealth + 1).
    """
    delta = params.loc[("delta", "delta"), "value"]
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else n_choices

    # In last period, nothing is saved for the next period (since there is none),
    # Hence, everything is consumed, c_T(M, d) = M
    for index, state in enumerate(choice_range):
        policy[n_periods - 1, index, 0, 1:] = copy.deepcopy(savings_grid)  # M
        policy[n_periods - 1, index, 1, 1:] = copy.deepcopy(savings_grid)  # c(M, d)

        value[n_periods - 1, index, 0, 2:] = (
            utility_func(policy[n_periods - 1, index, 0, 2:], params) - delta * state
        )
        value[n_periods - 1, index, 1, 2:] = (
            utility_func(policy[n_periods - 1, index, 1, 2:], params) - delta * state
        )
        value[n_periods - 1, index, :, :2] = 0

    return policy, value
