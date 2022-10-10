from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np


def solve_final_period(
    states: np.ndarray,
    savings_grid: np.ndarray,
    *,
    options: Dict[str, int],
    compute_utility: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes solution to final period for policy and value function.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        state_space (np.ndarray): Collection of all possible states.
        indexer (np.ndarray): Indexer object, that maps states to indexes.
        savings_grid (np.ndarray): Array of shape (n_wealth_grid,) denoting the
            exogenous savings grid.
        options (dict): Options dictionary.
        compute_utility (callable): Function for computation of agent's utility.

    Returns:
        (tuple): Tuple containing

        - policy (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific policy function; of shape
            [n_states, n_discrete_choices, 2, 1.1 * (n_grid_wealth + 1)].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each state and each discrete choice.
        - value (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_states, n_discrete_choices, 2, 1.1 * (n_grid_wealth + 1)].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each state and each discrete choice.
    """
    n_choices = options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else range(n_choices)
    n_states = states.shape[0]

    policy_final = np.empty(
        (n_states, n_choices, 2, int(1.1 * (len(savings_grid) + 1)))
    )
    value_final = np.empty((n_states, n_choices, 2, int(1.1 * (len(savings_grid) + 1))))
    policy_final[:] = np.nan
    value_final[:] = np.nan

    end_grid = len(savings_grid) + 1

    # In last period, nothing is saved for the next period (since there is none).
    # Hence, everything is consumed, c_T(M, d) = M
    for state_index in range(n_states):

        for index, choice in enumerate(choice_range):
            policy_final[state_index, index, :, 0] = 0
            policy_final[state_index, index, 0, 1:end_grid] = savings_grid  # M
            policy_final[state_index, index, 1, 1:end_grid] = savings_grid  # c(M, d)

            value_final[state_index, index, :, :2] = 0
            value_final[state_index, index, 0, 1:end_grid] = savings_grid

            # Start with second entry of savings grid to avaid taking the log of 0
            # (the first entry) when computing utility
            value_final[state_index, index, 1, 2:end_grid] = compute_utility(
                savings_grid[1:], choice
            )

    return policy_final, value_final
