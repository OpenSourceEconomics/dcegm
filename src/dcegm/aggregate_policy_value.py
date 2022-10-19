from typing import Callable

import numpy as np
import pandas as pd


def calc_expected_value(
    next_period_value: np.ndarray,
    params: pd.DataFrame,
) -> np.ndarray:
    """Computes the expected value of the next period.

    Args:
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].

    Returns:
        (np.ndarray): 1d array of the agent's expected value of the next period.
            Shape (n_grid_wealth,).
    """
    log_sum = _calc_log_sum(
        next_period_value, lambda_=params.loc[("shocks", "lambda"), "value"]
    )
    return log_sum


def calc_current_period_value(
    wealth: np.ndarray,
    next_period_value: np.ndarray,
    choice: int,
    beta: float,
    compute_utility: Callable,
) -> np.ndarray:
    """Compute the agent's value in the credit constrained region.

    Args:
        compute_utility (callable): User-defined function to compute the agent's
            utility. The input ``params``` is already partialled in.

    """
    utility = compute_utility(wealth, choice)
    value_constrained = utility + beta * next_period_value

    return value_constrained


def calc_next_period_choice_probs(
    next_period_value: np.ndarray,
    choice: int,
    taste_shock_scale: float,
) -> np.ndarray:
    """Calculates the probability of working in the next period.

    Args:
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        choice (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        taste_shock_scale (float): The taste shock scale.
    Returns:
        prob_working (np.ndarray): Probability of working next period. Array of
            shape (n_quad_stochastic * n_grid_wealth,).
    """
    col_max = np.amax(next_period_value, axis=0)
    next_period_value_ = next_period_value - col_max

    # Eq. (15), p. 334 IJRS (2017)
    choice_prob = np.exp(next_period_value_[choice, :] / taste_shock_scale) / np.sum(
        np.exp(next_period_value_ / taste_shock_scale), axis=0
    )

    return choice_prob


def _calc_log_sum(next_period_value: np.ndarray, lambda_: float) -> np.ndarray:
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

    # Eq. (14), p. 334 IJRS (2017)
    logsum = col_max + lambda_ * np.log(
        np.sum(np.exp((next_period_value_) / lambda_), axis=0)
    )

    return logsum
