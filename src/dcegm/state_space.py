"""Functions for creating the state space and the state choice space."""
from typing import Callable
from typing import Dict

import numpy as np


def get_child_states_index(
    state_space: np.ndarray,
    state_choice_space: np.ndarray,
    map_state_to_index: np.ndarray,
) -> np.ndarray:
    """Select state-specific child nodes. Will be a user defined function later.

    ToDo: We need to think about how to incorporate updating from state variables,
    e.g. experience.

    Args:
        state_space_admissible_choices (np.ndarray): 2d array with the collection of all
            states with admissible choices.
            Shape is (n_states * n_admissible_choices, n_state_variables + 1).
        state_indexer (np.ndarray): Indexer object that maps states to indexes.

    Returns:
        np.ndarray: 2d array of shape
            (n_state_specific_choices, n_state_specific_choices)
            containing all child nodes the agent can reach from her given state.

    """
    # n_periods = options["n_periods"]
    # n_choices = options["n_discrete_choices"]
    # n_exog_process = options["n_exog_processes"]

    # Exogenous processes are always on the last entry of the state space. Moreover, we
    # treat all of them as admissible in each period. If there exists an absorbing
    # state, this is reflected by a 0 percent transition probability.
    (n_periods, n_choices, n_exog_processes) = map_state_to_index.shape
    n_states_times_feasible_choices = state_choice_space.shape[0]

    n_states_without_period = state_space.shape[0] // n_periods  # 4

    indices_child_nodes = np.empty(
        (n_states_times_feasible_choices, n_exog_processes),
        dtype=int,
    )

    for idx in range(n_states_times_feasible_choices):
        state_choice_vec = state_choice_space[idx]
        period = state_choice_vec[0]
        state_vec = state_choice_vec[:-1]
        lagged_choice = state_choice_vec[-1]

        state_vec_next = state_vec.copy()
        state_vec_next[0] += 1  # Increment period

        if state_vec_next[0] < n_periods:
            state_vec_next[1] = lagged_choice

            for exog_process in range(n_exog_processes):
                state_vec_next[-1] = exog_process

                indices_child_nodes[idx, exog_process] = (
                    map_state_to_index[tuple(state_vec_next)]
                    - (period + 1) * n_states_without_period
                )

    return indices_child_nodes


def create_state_choice_space(
    state_space, map_state_to_index, get_state_specific_choice_set
):
    """Create a dictionary with all admissible choices for each state.

    Args:
        state_space (np.ndarray): Collection of all possible states.
        state_indexer (np.ndarray): Indexer object that maps states to indexes.
            The shape of this object quite complicated. For each state variable it
             has the number of possible states as "row", i.e.
            (n_poss_states_statesvar_1, n_poss_states_statesvar_2, ....)
        get_state_specific_choice_set (Callable): User-supplied function returning for
            each state all possible choices.

    Returns:
        np.ndarray: Array with the collection of all states with admissible choices.
            Shape is (num_states_admissible_choices, n_state_variables + 1).
        np.ndarray: Indexer object that maps state_id and choice to index of the
            state_space_admissible_choices

    """
    n_states, n_state_variables = state_space.shape
    n_periods, n_choices, n_exog_processes = map_state_to_index.shape

    state_choice_space = np.zeros(
        (n_states * n_choices, n_state_variables + 1),
        dtype=int,
    )
    # (n_states * n_choices, n_state_variables + 1)
    # here: n_states = n_periods + n_choices_lagged

    indexer_state_choice_space = np.full(
        (n_states, n_choices), dtype=int, fill_value=-99
    )

    idx = 0
    for state_idx in range(n_states):
        state_vec = state_space[state_idx]

        choice_set = get_state_specific_choice_set(
            state_vec, state_space, map_state_to_index
        )

        for feasible_choice in choice_set:
            state_choice_space[idx, :-1] = state_vec
            state_choice_space[idx, -1] = feasible_choice
            indexer_state_choice_space[state_idx, feasible_choice] = idx
            idx += 1

    return state_choice_space[:idx], indexer_state_choice_space


def get_feasible_choice_space(
    state_space: np.ndarray,
    map_states_to_indices: np.ndarray,
    get_state_specific_choice_set: Callable,
    options: Dict[str, int],
) -> np.ndarray:
    """Create binary array for storing the feasible choices for each state.

    Args:
        state_space (np.ndarray): Collection of all possible states.
        state_indexer (np.ndarray): 2d array of shape (n_periods, n_choices) containing
            the indexer object that maps states to indices in the state space.
        get_state_specific_choice_set (Callable): User-supplied function returning for
            each state all possible choices.
        options (dict): Options dictionary.

    Returns:
        np.ndarray: 2d array of shape (n_states, n_choices) indicating if choices
            are feasible.

    """
    n_choices = options["n_discrete_choices"]
    n_states = state_space.shape[0]

    choice_space = np.zeros((n_states, n_choices), dtype=int)

    for state_idx in range(n_states):
        state_vec = state_space[state_idx]

        choice_set = get_state_specific_choice_set(
            state_vec, state_space, map_states_to_indices
        )

        choice_space[state_idx, choice_set] = 1  # choice set is feasible

    return choice_space
