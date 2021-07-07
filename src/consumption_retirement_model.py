"""Model specific utility, wealth, and value functions."""
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy import interpolate


def utility_func_crra(
    current_consumption: np.ndarray, params: pd.DataFrame
) -> np.ndarray:
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        current_consumption (np.ndarray): Level of the agent's consumption in the
            current period. Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (np.ndarray): Array of agent's utility in the current period
            with shape (n_grid_wealth,).
    """
    theta = params.loc[("utility_function", "theta"), "value"]

    if theta == 1:
        utility = np.log(current_consumption)
    else:
        utility = (current_consumption ** (1 - theta) - 1) / (1 - theta)

    return utility


def marginal_utility_crra(
    current_consumption: np.ndarray, params: pd.DataFrame
) -> np.ndarray:
    """Computes marginal utility of CRRA utility function.

    Args:
        current_consumption (np.ndarray): Level of the agent's consumption in the
            curent period. Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
            Relevant here is the CRRA coefficient theta.

    Returns:
        marginal_utility (np.ndarray): Marginal utility of CRRA consumption
            function with shape (n_grid_wealth,).
    """
    theta = params.loc[("utility_function", "theta"), "value"]
    marginal_utility = current_consumption ** (-theta)

    return marginal_utility


def inverse_marginal_utility_crra(
    marginal_utility: np.ndarray, params: pd.DataFrame,
) -> np.ndarray:
    """Computes the inverse marginal utility of a CRRA utility function.

    Args:
        marginal_utility (np.ndarray): Level of marginal CRRA utility.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
    Returns:
        inverse_marginal_utility(np.ndarray): Inverse of the marginal utility of
            a CRRA consumption function.
    """
    theta = params.loc[("utility_function", "theta"), "value"]
    inverse_marginal_utility = marginal_utility ** (-1 / theta)

    return inverse_marginal_utility


def compute_current_period_consumption(
    next_period_marginal_utility: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    matrix_marginal_wealth: np.ndarray,
    quad_weights: np.ndarray,
    params: pd.DataFrame,
    inv_marginal_utility_func: Callable,
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
        current_period_consumption (np.ndarray): Consumption in the current
            period. Array of shape (n_grid_wealth,).
    """
    beta = params.loc[("beta", "beta"), "value"]

    # RHS of Euler Eq., p. 337 IJRS (2017)
    # Integrate out uncertainty over stochastic income y
    rhs_euler = _get_rhs_euler(
        next_period_marginal_utility,
        matrix_next_period_wealth,
        matrix_marginal_wealth,
        quad_weights,
    )

    current_period_consumption = inv_marginal_utility_func(beta * rhs_euler, params)

    return current_period_consumption


def _get_rhs_euler(
    next_period_marginal_utility: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    matrix_marginal_wealth: np.ndarray,
    quad_weights: np.ndarray,
) -> np.ndarray:
    """Computes the right-hand side of the Euler equation, p. 337 IJRS (2017).

    Args:
        next_period_marginal_utility (np.ndarray): Array of next period's
            marginal utility of shape (n_quad_stochastic * n_grid_wealth,).
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


def compute_value_function(
    next_period: int,
    state: int,
    value: List[np.ndarray],
    next_period_wealth_matrix: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_function: Callable,
) -> np.ndarray:
    """Computes the value function of the next period t+1.

    Take into account credit-constrained regions.
    Use interpolation in non-constrained region and apply extrapolation
    where the observed wealth exceeds the maximum wealth level.

    Args:
        next_period (int): Next period, t+1.
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which we set
            to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the array contains the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice.
        next_period_wealth_matrix (np.ndarray): Array of of all possible next
            period wealths. Shape (n_quad_stochastic, n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        uility_func (callable): Utility function.

    Returns:
        value_function (np.ndarray): Value function. Array of shape
            (n_quad_stochastic * n_grid_wealth,).
    """
    delta = params.loc[("delta", "delta"), "value"]
    beta = params.loc[("beta", "beta"), "value"]

    # If only one state, i.e. no discrete choices to make,
    # set state index to 0
    state_index = 0 if options["n_discrete_choices"] < 2 else state

    next_period_wealth = next_period_wealth_matrix.flatten("F")

    value_function = np.full(next_period_wealth.shape, np.nan)

    # Mark credit constrained region
    constrained_region = (
        next_period_wealth < value[next_period][state_index][0, 1]
    )  # Last dim denotes grid point j=1

    # Calculate t+1 value function in constrained region
    value_function[constrained_region] = (
        utility_function(next_period_wealth[constrained_region], params)
        - state * delta
        + beta * value[next_period][state_index][1, 0]
    )

    # Calculate t+1 value function in non-constrained region
    # via inter- and extrapolation
    value_function_interp = interpolate.interp1d(
        x=value[next_period][state_index][0, :],  # endogenous wealth grid
        y=value[next_period][state_index][1, :],  # value_function
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )
    value_function[~constrained_region] = value_function_interp(
        next_period_wealth[~constrained_region]
    )

    return value_function


def compute_next_period_marginal_utility(
    period: int,
    state: int,
    policy: List[np.ndarray],
    matrix_next_period_wealth: np.ndarray,
    next_period_value: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> np.ndarray:
    """Computes the marginal utility of the next period.
    
    Args:
        period (int): Current period t.
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            concurrent local optima for consumption. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which we set
            to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the arrays contain the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the (consumption) policy 
            function c(M, d), for each time period and each discrete choice.
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
        next_period_marg_util (np.ndarray): Array of next period's
            marginal utility of shape (n_quad_stochastic, n_grid_wealth,).
    """
    n_choices = options["n_discrete_choices"]

    next_period_consumption = _calc_next_period_consumption(
        period, policy, matrix_next_period_wealth, options
    )

    # If no discrete alternatives, only one state, i.e. one column with index = 0
    if n_choices < 2:
        next_period_marg_util = marginal_utility_crra(
            next_period_consumption[0, :], params
        )
    else:
        prob_working = _calc_next_period_choice_probs(
            next_period_value, state, params, options
        )

        next_period_marg_util = prob_working * marginal_utility_crra(
            next_period_consumption[1, :], params
        ) + (1 - prob_working) * marginal_utility_crra(
            next_period_consumption[0, :], params
        )

    return next_period_marg_util


def compute_expected_value(
    state: int,
    next_period_value: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    quad_weights: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> np.ndarray:
    """Computes the expected value of the next period.

    Args:
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        quad_weights (np.ndarray): Weights associated with the stochastic
            quadrature points of shape (n_quad_stochastic,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
        expected_value (np.ndarray): Array of current period's expected value of
            next_period. Shape (n_grid_wealth,).
    """
    # Taste shock (scale) parameter
    lambda_ = params.loc[("shocks", "lambda"), "value"]

    # If no discrete alternatives, only one state and logsum is not needed
    state_index = 0 if options["n_discrete_choices"] < 2 else state

    if state_index == 1:
        # Continuation value of working
        expected_value = np.dot(
            quad_weights.T,
            _calc_logsum(next_period_value, lambda_).reshape(
                matrix_next_period_wealth.shape, order="F"
            ),
        )
    else:
        expected_value = np.dot(
            quad_weights.T,
            next_period_value[0, :].reshape(matrix_next_period_wealth.shape, order="F"),
        )

    return expected_value


def _calc_next_period_consumption(
    period: int,
    policy: List[np.ndarray],
    matrix_next_period_wealth: np.ndarray,
    options: Dict[str, int],
) -> np.ndarray:
    """Computes consumption in the next period via linear interpolation.

    Extrapolate lineary in wealth regions to larger than max_wealth.

    Args:
        period (int): Current period t.
        policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            concurrent local optima for consumption. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which we set
            to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the arrays contain the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the (consumption) policy 
            function c(M, d), for each time period and each discrete choice.
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        options (dict): Options dictionary.

    Returns:
        next_period_consumption_interp (np.ndarray): Array of next period
            consumption of shape (n_quad_stochastic * n_grid_wealth,).
    """
    n_choices = options["n_discrete_choices"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]
    choice_range = [0] if n_choices < 2 else range(n_choices)

    next_period_consumption_interp = np.empty(
        (n_choices, n_quad_stochastic * n_grid_wealth)
    )

    for state_index in choice_range:
        next_period_wealth = policy[period + 1][state_index][0, :]
        next_period_consumption = policy[period + 1][state_index][1, :]

        interpolation_func = interpolate.interp1d(
            next_period_wealth,
            next_period_consumption,
            bounds_error=False,
            fill_value="extrapolate",
            kind="linear",
        )
        next_period_consumption_interp[state_index, :] = interpolation_func(
            matrix_next_period_wealth
        ).flatten("F")

    return next_period_consumption_interp


def _calc_next_period_choice_probs(
    next_period_value: np.ndarray,
    state: int,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> np.ndarray:
    """Calculates the probability of working in the next period.
    
    Args:
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
        prob_working (np.ndarray): Probability of working next period. Array of
            shape (n_quad_stochastic * n_grid_wealth,).
    """
    # Taste shock (scale) parameter
    lambda_ = params.loc[("shocks", "lambda"), "value"]

    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]

    if state == 1:
        col_max = np.amax(next_period_value, axis=0)
        next_period_value_ = next_period_value - col_max

        # eq. (15), p. 334
        prob_working = np.exp(next_period_value_[state, :] / lambda_) / np.sum(
            np.exp(next_period_value_ / lambda_), axis=0
        )

    else:
        prob_working = np.zeros(n_quad_stochastic * n_grid_wealth)

    return prob_working


def _calc_logsum(next_period_value: np.ndarray, lambda_: float) -> np.ndarray:
    """Calculates the log-sum needed for computing the expected value function.

    The log-sum formula may also be referred to as the 'smoothed max function',
    see eq. (50), p. 335 (Appendix).
    
    Args:
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        lambda_ (float): Taste shock (scale) parameter.

    Returns:
        logsum (np.ndarray): Log-sum formula inside the expected value function.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
    """
    col_max = np.amax(next_period_value, axis=0)
    next_period_value_ = next_period_value - col_max

    # eq. (14), p. 334
    logsum = col_max + lambda_ * np.log(
        np.sum(np.exp((next_period_value_) / lambda_), axis=0)
    )

    return logsum
