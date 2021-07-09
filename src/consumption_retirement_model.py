"""Model specific utility, wealth, and value functions."""
from typing import Dict

import numpy as np
import pandas as pd


def compute_value_function(
    state: int,
    current_consumption: np.ndarray,
    next_period_value: np.ndarray,
    params: pd.DataFrame,
) -> np.ndarray:
    """Computes the agent's value function in the current (not her final) period.
    
    Args:
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        current_consumption (np.ndarray): Level of the agent's consumption in the
            current period. Array of (i) shape (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].

    Returns:
        value (np.ndarray): Values of the value function in the current (not the
            final) period. Array of shape (i) (n_quad_stochastic * n_grid_wealth,) or 
            (ii) (n_grid_wealth,), depending on where :func:`compute_value_function`
            is called (see above).
    """
    delta = params.loc[("delta", "delta"), "value"]
    beta = params.loc[("beta", "beta"), "value"]

    utility = utility_func_crra(current_consumption, params)
    value = utility - state * delta + beta * next_period_value

    return value


def compute_value_function_final_period(
    state: int, current_consumption: np.ndarray, params: pd.DataFrame,
) -> np.ndarray:
    """Computes the agent's value function in the final period of her life cycle.
    
    Args:
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        current_consumption (np.ndarray): Level of the agent's consumption in the
            current period. Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].

    Returns:
        value_function_final_period (np.ndarray): Values of the value function
            in the agent's final period. Array of shape 
            (n_quad_stochastic * n_grid_wealth,).
    """
    delta = params.loc[("delta", "delta"), "value"]

    utility = utility_func_crra(current_consumption, params,).flatten("F")
    value_final_period = utility - state * delta

    return value_final_period


def utility_func_crra(
    current_consumption: np.ndarray, params: pd.DataFrame
) -> np.ndarray:
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        current_consumption (np.ndarray): Level of the agent's consumption in the
            current period. Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
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
            Array of shape (n_grid_wealth,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
    Returns:
        inverse_marginal_utility(np.ndarray): Inverse of the marginal utility of
            a CRRA consumption function.
    """
    theta = params.loc[("utility_function", "theta"), "value"]
    inverse_marginal_utility = marginal_utility ** (-1 / theta)

    return inverse_marginal_utility


def compute_next_period_marginal_utility(
    state: int,
    next_period_consumption: np.ndarray,
    next_period_value: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> np.ndarray:
    """Computes the marginal utility of the next period.
    
    Args:
        period (int): Current period t.
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        next_period_consumption (np.ndarray): Array of next period consumption 
            of shape (n_choices, n_quad_stochastic * n_grid_wealth). Contains 
            interpolated values.
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
        next_period_marg_util (np.ndarray): Array of next period's
            marginal utility of shape (n_quad_stochastic * n_grid_wealth,).
    """
    n_choices = options["n_discrete_choices"]

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
    matrix_next_period_wealth: np.ndarray,
    next_period_value: np.ndarray,
    quad_weights: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> np.ndarray:
    """Computes the expected value of the next period.

    Args:
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        quad_weights (np.ndarray): Weights associated with the stochastic
            quadrature points of shape (n_quad_stochastic,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
        expected_value (np.ndarray): Expected value of next period. Array of
            shape (n_grid_wealth,).
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

        # Eq. (15), p. 334 IJRS (2017
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

    # Eq. (14), p. 334 IJRS (2017
    logsum = col_max + lambda_ * np.log(
        np.sum(np.exp((next_period_value_) / lambda_), axis=0)
    )

    return logsum
