from typing import Dict
from typing import Tuple

import numpy as np


def create_state_space(options: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create state space objects and indexer.

    Args:
        options (dict): Options dictionary.

    Returns:
        states (np.ndarray): Collection of all possible states of shape
            (n_periods * n_choices, n_choices).
        indexer (np.ndarray): Indexer object that maps states to indexes.
            Shape (n_periods, n_choices).

    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]

    shape = (n_periods, n_choices)
    indexer = np.full(shape, -9999, dtype=np.int64)

    _state_space = []

    i = 0
    for period in range(n_periods):
        for last_period_decision in range(n_choices):
            indexer[period, last_period_decision] = i

            row = [period, last_period_decision]
            _state_space.append(row)

            i += 1

    state_space = np.array(_state_space, dtype=np.int64)

    return state_space, indexer


def get_state_choice_set(
    state: np.ndarray,
    state_space: np.ndarray,
    indexer: np.ndarray,
) -> np.ndarray:
    """Select choice set per state. Will be a user defined function later.
    This is very basic in Ishakov.

    Args:
        state (np.ndarray): Current individual state.
        state_space (np.ndarray): Collection of all possible states.
        indexer (np.ndarray): Indexer object, that maps states to indexes.

    Returns:
        choice_set (np.ndarray): This is the choice set in this state.

    """
    if state[1] == 1:
        return np.array([1])
    else:
        return np.array(range(indexer.shape[1]))


def get_child_states(
    state: np.ndarray,
    state_space: np.ndarray,
    indexer: np.ndarray,
) -> np.ndarray:
    """Select child nodes set per state. Will be a user defined function later.

    Args:
        state (np.ndarray): Current individual state.
        state_space (np.ndarray): Collection of all possible states.
        indexer (np.ndarray): Indexer object, that maps states to indexes.

    Returns:
        child_nodes (np.ndarray): This is the choice set in this state.

    """
    # Child nodes are so far num_choices by state_space variables.
    choice_set_state = get_state_choice_set(state, state_space, indexer)
    child_nodes = np.empty((choice_set_state.shape[0], state_space.shape[1]), dtype=int)
    for i, choice in enumerate(choice_set_state):
        child_nodes[i, :] = state_space[indexer[state[0] + 1, choice]]
    return child_nodes


def get_index_by_state(state, indexer):
    """Can we get a better function than this?

    Args:
        state:
        indexer:

    Returns:

    """
    sub_ind = indexer[state[0]]
    for state_var in state[1:]:
        sub_ind = sub_ind[state_var]
    return sub_ind
