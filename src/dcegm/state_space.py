"""Functions for creating internal state space objects."""
import numpy as np


def get_map_from_state_to_child_nodes(
    state_space: np.ndarray,
    state_choice_space: np.ndarray,
    map_state_to_index: np.ndarray,
) -> np.ndarray:
    """Create indexer array that maps states to state-specific child nodes.

    Will be a user defined function later.

    ToDo: We need to think about how to incorporate updating from state variables,
    e.g. experience.

    Args:
        state_space (np.ndarray): 2d array of shape (n_states, n_state_variables + 1)
            which serves as a collection of all possible states. By convention,
            the first column must contain the period and the last column the
            exogenous processes. Any other state variables are in between.
            E.g. if the two state variables are period and lagged choice and all choices
            are admissible in each period, the shape of the state space array is
            (n_periods * n_choices, 3).
        state_choice_space (np.ndarray): 2d array of shape
            (n_feasible_states, n_state_and_exog_variables + 1) containing all feasible
            state-choice combinations.
        map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
            The shape of this object is quite complicated. For each state variable it
            has the number of possible states as rows, i.e.
            (n_poss_states_state_var_1, n_poss_states_state_var_2, ....).

    Returns:
        np.ndarray: 2d array of shape
            (n_feasible_state_choice_combs, n_choices * n_exog_processes)
            containing indices of all child nodes the agent can reach
            from any given state.

    """
    # n_periods = options["n_periods"]
    # n_choices = options["n_discrete_choices"]
    # n_exog_process = options["n_exog_processes"]

    # Exogenous processes are always on the last entry of the state space. Moreover, we
    # treat all of them as admissible in each period. If there exists an absorbing
    # state, this is reflected by a 0 percent transition probability.
    n_periods, n_choices, n_exog_processes = map_state_to_index.shape
    n_feasible_state_choice_combs = state_choice_space.shape[0]

    n_states_over_periods = state_space.shape[0] // n_periods

    map_state_to_child_nodes = np.empty(
        (n_feasible_state_choice_combs, n_exog_processes),
        dtype=int,
    )

    for idx in range(n_feasible_state_choice_combs):
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

                map_state_to_child_nodes[idx, exog_process] = (
                    map_state_to_index[tuple(state_vec_next)]
                    - (period + 1) * n_states_over_periods
                )

    return map_state_to_child_nodes


def create_state_choice_space(
    state_space, map_state_to_index, get_state_specific_choice_set
):
    """Create state choice space of all feasible state-choice combinations.

    Also conditional on any realization of exogenous processes.

    Args:
        state_space (np.ndarray): 2d array of shape (n_states, n_state_variables + 1)
            which serves as a collection of all possible states. By convention,
            the first column must contain the period and the last column the
            exogenous processes. Any other state variables are in between.
            E.g. if the two state variables are period and lagged choice and all choices
            are admissible in each period, the shape of the state space array is
            (n_periods * n_choices, 3).
        map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
            The shape of this object is quite complicated. For each state variable it
            has the number of possible states as rows, i.e.
            (n_poss_states_state_var_1, n_poss_states_state_var_2, ....).
        get_state_specific_choice_set (Callable): User-supplied function that returns
            the set of feasible coices for a given state.

    Returns:
        np.ndarray: 2d array of shape
            (n_feasible_states, n_state_and_exog_variables + 1) containing all
            feasible state-choice combinations. By convention, the second to last
            column contains the exogenous process. The last column always contains the
            choice to be made (which is not a state variable).

    """
    n_states, n_state_and_exog_variables = state_space.shape
    _n_periods, n_choices, _n_exog_processes = map_state_to_index.shape

    state_choice_space = np.zeros(
        (n_states * n_choices, n_state_and_exog_variables + 1),
        dtype=int,
    )
    sum_state_choices_to_state = np.zeros((n_states, n_states * n_choices), dtype=int)

    map_state_choice_to_state = np.zeros((n_states * n_choices), dtype=int)

    idx = 0
    for state_idx in range(n_states):
        state_vec = state_space[state_idx]

        choice_set = get_state_specific_choice_set(
            state_vec, state_space, map_state_to_index
        )

        for feasible_choice in choice_set:
            state_choice_space[idx, :-1] = state_vec
            state_choice_space[idx, -1] = feasible_choice
            sum_state_choices_to_state[state_idx, idx] = 1
            map_state_choice_to_state[idx] = state_idx
            idx += 1

    return (
        state_choice_space[:idx],
        sum_state_choices_to_state[:, :idx],
        map_state_choice_to_state[:idx],
    )
