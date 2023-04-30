from typing import Callable

import numpy as np


def get_child_states(
    state: np.ndarray,
    state_space: np.ndarray,
    indexer: np.ndarray,
    get_state_specific_choice_set: Callable,
) -> np.ndarray:
    """Select state-specific child nodes. Will be a user defined function later.

    ToDo: We need to think about how to incorporate updating from state variables,
    e.g. experience.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) defining the agent's
            state. In Ishkakov, an agent's state is defined by her (i) age (i.e. the
            current period) and (ii) her lagged labor market choice.
            Hence n_state_variables = 2.
        state_space (np.ndarray): Collection of all possible states.
        indexer (np.ndarray): 2d array of shape (n_periods, n_choices) containing
            the indexer object that maps states to indices in the state space.
        get_state_specific_choice_set (Callable): User-supplied function returning for
            each state all possible choices.

    Returns:
        np.ndarray: 2d array of shape
            (n_state_specific_choices, n_state_specific_choices)
            containing all child nodes the agent can reach from her given state.

    """
    # Exogenous processes are always on the last entry of the state space. Moreover, we
    # treat all of them as admissible in each period. If there exists a absorbing state,
    # this is reflected by a 0 percent transition probability.
    n_exog_processes = indexer.shape[-1]

    # Get all admissible choices.
    state_specific_choice_set = get_state_specific_choice_set(
        state, state_space, indexer
    )

    child_nodes = np.empty(
        (indexer.shape[1], n_exog_processes, state_space.shape[1]),
        dtype=int,
    )  # (n_admissible_choices, n_exog_processes, n_state_variables)
    new_state = state.copy()
    new_state[0] += 1
    for choice in state_specific_choice_set:
        new_state[1] = choice
        for exog_proc_state in range(n_exog_processes):
            new_state[-1] = exog_proc_state
            child_nodes[choice, exog_proc_state, :] = state_space[
                indexer[tuple(new_state)]
            ]

    return child_nodes


def get_child_indexes(
    state: np.ndarray,
    state_space: np.ndarray,
    indexer: np.ndarray,
    get_state_specific_choice_set: Callable,
) -> np.ndarray:
    """Create array of child indexes.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) defining the agent's
            state. In Ishkakov, an agent's state is defined by her (i) age (i.e. the
            current period) and (ii) her lagged labor market choice.
            Hence n_state_variables = 2.
        state_space (np.ndarray): Collection of all possible states.
        indexer (np.ndarray): 2d array of shape (n_periods, n_choices) containing
            the indexer object that maps states to indices in the state space.
        get_state_specific_choice_set (Callable): User-supplied function returning for
            each state all possible choices.

    Returns:
        np.ndarray: 2d array of shape
            (n_state_specific_choices, n_state_specific_choices)
            containing all child nodes the agent can reach from her given state.

    """
    child_states = get_child_states(
        state, state_space, indexer, get_state_specific_choice_set
    )

    child_indexes = np.full(
        (child_states.shape[0], child_states.shape[1]), fill_value=-99, dtype=int
    )
    for choice_ind in range(child_states.shape[0]):
        for exog_proc_ind in range(child_states.shape[1]):
            child_indexes[choice_ind, exog_proc_ind] = indexer[
                tuple(child_states[choice_ind, exog_proc_ind, :])
            ]
    return child_indexes
