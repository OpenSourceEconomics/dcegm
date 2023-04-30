from typing import Callable
from typing import Dict

import numpy as np


def get_child_states_index(
    state_space_admissible_choices: np.ndarray,
    state_indexer: np.ndarray,
) -> np.ndarray:
    """Select state-specific child nodes. Will be a user defined function later.

    ToDo: We need to think about how to incorporate updating from state variables,
    e.g. experience.

    Args:
        state_space_admissible_choices (np.ndarray): Array with the collection of all
            states with admissible choices.
            Shape is (num_states_admissible_choices, n_state_variables + 1).
        state_indexer (np.ndarray): Indexer object that maps states to indexes.

    Returns:
        np.ndarray: 2d array of shape
            (n_state_specific_choices, n_state_specific_choices)
            containing all child nodes the agent can reach from her given state.

    """
    # Exogenous processes are always on the last entry of the state space. Moreover, we
    # treat all of them as admissible in each period. If there exists a absorbing state,
    # this is reflected by a 0 percent transition probability.
    n_exog_processes = state_indexer.shape[-1]
    num_states_admissible_choices = state_space_admissible_choices.shape[0]

    child_nodes = np.empty(
        (num_states_admissible_choices, n_exog_processes),
        dtype=int,
    )  # (n_admissible_choices, n_exog_processes, n_state_variables)

    for id_state_choice in range(num_states_admissible_choices):
        state_choice = state_space_admissible_choices[id_state_choice]
        state = state_choice[:-1]
        choice = state_choice[-1]
        new_state = state.copy()
        new_state[0] += 1
        if new_state[0] < state_indexer.shape[0]:
            new_state[1] = choice
            for exog_proc_state in range(n_exog_processes):
                new_state[-1] = exog_proc_state
                child_nodes[id_state_choice, exog_proc_state] = state_indexer[
                    tuple(new_state)
                ]

    return child_nodes


def create_state_space_admissible_choices(
    state_space, state_indexer, get_state_specific_choice_set
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
    state_space_admissible_choices = np.zeros(
        (state_space.shape[0] * state_indexer.shape[1], state_space.shape[1] + 1),
        dtype=int,
    )
    indexer_states_admissible_choices = np.full(
        (state_space.shape[0], state_indexer.shape[1]), dtype=int, fill_value=-99
    )
    counter = 0
    for index in range(state_space.shape[0]):
        state = state_space[index]
        choice_set = get_state_specific_choice_set(state, state_space, state_indexer)
        for choice in choice_set:
            state_space_admissible_choices[counter, :-1] = state
            state_space_admissible_choices[counter, -1] = choice
            indexer_states_admissible_choices[index, choice] = counter
            counter += 1
    return state_space_admissible_choices[:counter], indexer_states_admissible_choices


def get_possible_choices_array(
    state_space: np.ndarray,
    state_indexer: np.ndarray,
    get_state_specific_choice_set: Callable,
    options: Dict[str, int],
) -> np.ndarray:
    """Create binary array for storing the possible choices for each state.

    Args:
        state_space (np.ndarray): Collection of all possible states.
        state_indexer (np.ndarray): 2d array of shape (n_periods, n_choices) containing
            the indexer object that maps states to indices in the state space.
        get_state_specific_choice_set (Callable): User-supplied function returning for
            each state all possible choices.
        options (dict): Options dictionary.

    Returns:
        np.ndarray: Binary 2d array of shape (n_states, n_choices)
            indicating if choice is possible.

    """
    n_choices = options["n_discrete_choices"]
    choices_array = np.zeros((state_space.shape[0], n_choices), dtype=int)
    for index in range(state_space.shape[0]):
        state = state_space[index]
        choice_set = get_state_specific_choice_set(state, state_space, state_indexer)
        choices_array[index, choice_set] = 1

    return choices_array
