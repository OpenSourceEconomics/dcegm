"""User-supplied function for the final period."""
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np


def solve_final_period_scalar(
    state_vec: np.ndarray,  # noqa: U100
    choice: int,
    begin_of_period_resources: float,
    params: Dict[str, float],
    options: Dict[str, int],  # noqa: U100
    compute_utility: Callable,
    compute_marginal_utility: Callable,
) -> Tuple[float, float]:
    """Compute optimal consumption policy and value function in the final period.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) containing the
            period-specific state vector.
        choice (int): The agent's choice in the current period.
        begin_of_period_resources (float): The agent's begin of period resources.
        compute_utility (callable): Function for computation of agent's utility.
        compute_marginal_utility (callable): Function for computation of agent's
        params (dict): Dictionary of model parameters.
        options (dict): Options dictionary.

    Returns:
        tuple:

        - consumption (float): The agent's consumption in the final period.
        - value (float): The agent's value in the final period.
        - marginal_utility (float): The agent's marginal utility .

    """
    marginal_utility = compute_marginal_utility(
        consumption=begin_of_period_resources, params=params
    )
    value = compute_utility(
        consumption=begin_of_period_resources, choice=choice, params=params
    )
    consumption = begin_of_period_resources

    return marginal_utility, value, consumption
