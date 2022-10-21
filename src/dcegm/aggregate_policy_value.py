from typing import Callable

import numpy as np


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
