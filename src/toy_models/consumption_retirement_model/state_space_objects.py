from typing import Dict
from typing import Tuple

import numpy as np


def create_state_space(options: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create state space object and indexer.

    Args:
        options (dict): Options dictionary.

    Returns:
        state_space (np.ndarray): Collection of all possible states of shape
            (n_periods * n_choices, n_choices).
        indexer (np.ndarray): Indexer object that maps states to indexes.
            Shape (n_periods, n_choices).

    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]
    n_exog_process = options["n_exog_processes"]

    shape = (n_periods, n_choices, n_exog_process)
    indexer = np.full(shape, -9999, dtype=np.int64)

    _state_space = []

    i = 0
    for period in range(n_periods):
        for last_period_decision in range(n_choices):
            for exog_process in range(n_exog_process):

                indexer[period, last_period_decision, exog_process] = i

                row = [period, last_period_decision, exog_process]
                _state_space.append(row)

                i += 1

    state_space = np.array(_state_space, dtype=np.int64)

    return state_space, indexer


def get_state_specific_choice_set(
    state: np.ndarray,
    state_space: np.ndarray,
    indexer: np.ndarray,
) -> np.ndarray:
    """Select state-specific choice set. Will be a user defined function later.

    This is very basic in Ishkakov.

    Args:
        state (np.ndarray): Array of shape (n_state_variables,) defining the agent's
            state. In Ishkakov, an agent's state is defined by her (i) age (i.e. the
            current period) and (ii) her lagged labor market choice.
            Hence n_state_variables = 2.
        state_space (np.ndarray): Collection of all possible states of shape
            (n_periods * n_choices, n_choices).
        indexer (np.ndarray): Indexer object that maps states to indexes.
            Shape (n_periods, n_choices).

    Returns:
        choice_set (np.ndarray): The agent's (restricted) choice set in the given
            state of shape (n_admissible_choices,).

    """
    n_state_variables = indexer.shape[1]

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if state[1] == 1:
        choice_set = np.array([1])
    else:
        choice_set = np.arange(n_state_variables)

    return choice_set
