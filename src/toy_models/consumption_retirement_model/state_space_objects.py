"""User-defined functions for creating state space objects."""
from typing import Dict
from typing import Tuple

import numpy as np


def create_state_space(options: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create state space object and indexer.

    We need to add the convention for the state space objects.

    Args:
        options (dict): Options dictionary.

    Returns:
        tuple:

        - state_space (np.ndarray): 2d array of shape
            (n_states, n_state_variables) which contains all possible states.
            The first column contains the period, the second column the lagged choice,
            and the third column the exogenous process.
        - indexer (np.ndarray): Indexer object that maps states to indexes. The shape of
            this object quite complicated. For each state variable it has the number of
            feasible states as "row", i.e.
            (n_poss_states_statesvar_1, n_poss_states_statesvar_2, ....)

    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]  # lagged_choice is a state variable
    n_exog_process = options["n_exog_processes"]

    shape = (n_periods, n_choices, n_exog_process)

    indexer = np.full(shape, -9999, dtype=np.int64)
    _state_space = []

    i = 0
    for period in range(n_periods):
        for choice in range(n_choices):
            for exog_process in range(n_exog_process):
                indexer[period, choice, exog_process] = i

                row = [period, choice, exog_process]
                _state_space.append(row)

                i += 1

    state_space = np.array(_state_space, dtype=np.int64)

    return state_space, indexer


def get_state_specific_choice_set(
    state: np.ndarray,
    state_space: np.ndarray,  # noqa: U100
    indexer: np.ndarray,
) -> np.ndarray:
    """Select state-specific choice set. Will be a user defined function later.

    This is very basic in Ishkakov et al (2017).

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
        choice_set (np.ndarray): 1d array of length (n_feasible_choices,) with the
            agent's (restricted) feasible choice set in the given state.

    """
    n_state_variables = indexer.shape[1]

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if state[1] == 1:
        choice_set = np.array([1])
    else:
        choice_set = np.arange(n_state_variables)

    # breakpoint()

    return choice_set
