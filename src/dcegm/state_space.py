from typing import Callable

import numpy as np


def get_child_states(
    state: np.ndarray,
    state_space: np.ndarray,
    indexer: np.ndarray,
    get_state_specific_choice_set: Callable,
) -> np.ndarray:
    """Select state-specific child nodes. Will be a user defined function later.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) defining the agent's
            state. In Ishkakov, an agent's state is defined by her (i) age (i.e. the
            current period) and (ii) her lagged labor market choice.
            Hence n_state_variables = 2.
        states (np.ndarray): Collection of all possible states of shape
            (n_periods * n_choices, n_choices).
        indexer (np.ndarray): 2d array of shape (n_periods, n_choices) containing
            the indexer object that maps states to indices in the state space.

    Returns:
        child_nodes (np.ndarray): 2d array of shape
            (n_state_specific_choices, n_state_specific_choices) containing all child
            nodes the agent can reach from her given state.

    """
    # Child nodes are so far n_choices by state_space variables.
    state_specific_choice_set = get_state_specific_choice_set(
        state, state_space, indexer
    )
    child_nodes = np.empty(
        (state_specific_choice_set.shape[0], state_space.shape[1]), dtype=int
    )  # (n_admissible_choices, n_state_variables)

    for i, choice in enumerate(state_specific_choice_set):
        child_nodes[i, :] = state_space[indexer[state[0] + 1, choice]]

    return child_nodes
