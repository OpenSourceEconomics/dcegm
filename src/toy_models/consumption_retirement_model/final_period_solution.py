"""User-supplied function for the final period."""
from collections.abc import Callable

import numpy as np


def solve_final_period_scalar(
    state: np.ndarray,
    begin_of_period_resources: float,
    choice: int,
    options: dict[str, int],
    params_dict: dict,
    compute_utility: Callable,
    compute_marginal_utility: Callable,
) -> tuple[float, float]:
    """Compute optimal consumption policy and value function in the final period.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        state (np.ndarray): Collection of all possible states. Shape is (n_states,).
        begin_of_period_resources (float): The agent's begin of period resources.
        choice (int): The agent's choice.
        options (dict): Options dictionary.
        params_dict (dict): Dictionary of parameters.
        compute_utility (callable): Function for computation of agent's utility.
        compute_marginal_utility (callable): Function for computation of agent's

    Returns:
        tuple:

        - consumption (float): The agent's consumption in the final period.
        - value (float): The agent's value in the final period.
        - marginal_utility (float): The agent's marginal utility .

    """
    consumption = begin_of_period_resources
    value = compute_utility(begin_of_period_resources, choice)
    marginal_utility = compute_marginal_utility(begin_of_period_resources)

    return consumption, value, marginal_utility
