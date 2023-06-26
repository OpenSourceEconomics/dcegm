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
    state_times_state_choice_mat = np.zeros((n_states, n_choices), dtype=int)

    idx = 0
    current_period = state_space[0, 0]
    # Ensure that states are ordered.
    period_min_idx = -1
    for state_idx in range(n_states):
        state_vec = state_space[state_idx]

        if current_period == state_vec[0]:
            period_min_idx = idx
            current_period += 1
        choice_set = get_state_specific_choice_set(
            state_vec, state_space, map_state_to_index
        )

        for choice in choice_set:
            state_choice_space[idx, :-1] = state_vec
            state_choice_space[idx, -1] = choice
            sum_state_choices_to_state[state_idx, idx] = 1
            map_state_choice_to_state[idx] = state_idx
            state_times_state_choice_mat[state_idx, choice] = idx - period_min_idx
            idx += 1
        # Fill up matrix with some state_choice index from the state, as we only use
        # this matrix to get the maximum across state_choice values and two times the
        # same value doesn't change the maximum. Only used in aggregation function.
        for choice in range(n_choices):
            if choice not in choice_set:
                state_times_state_choice_mat[state_idx, choice] = (
                    idx - 1 - period_min_idx
                )

    return (
        state_choice_space[:idx],
        sum_state_choices_to_state[:, :idx],
        map_state_choice_to_state[:idx],
        state_times_state_choice_mat,
    )


def select_period_objects(
    period,
    state_space,
    state_choice_space,
    sum_state_choices_to_state,
    map_state_choice_to_state,
    state_times_state_choice_mat,
    resources_beginning_of_period,
):
    """Select objects for the current period.

    Args:
        period (int): Current period.
        state_space (np.ndarray): 2d array of shape (n_states, n_periods) containing
            the state space.
        state_choice_space (np.ndarray): 2d array of shape (n_state_choices, n_states + 1)
            containing the state choice space.
        sum_state_choices_to_state (np.ndarray): 2d array of shape (n_states, n_state_choices)
            containing the mapping from state choices to states.
        map_state_choice_to_state (np.ndarray): 1d array of shape (n_state_choices,)
            containing the mapping from state choices to states.
        state_times_state_choice_mat (np.ndarray): 2d array of shape (n_states, n_state_choices)
            containing the mapping from states to state choices.
        resources_beginning_of_period (np.ndarray): 3d array of shape
            (n_states, n_exog_savings, n_stochastic_quad_points) containing the resources
            at the beginning of the current period.

    Returns:
        tuple:

        - idx_states_current_period (np.ndarray): 1d array of shape
            (n_states_current_period,).
        - idx_state_choices_current_period (np.ndarray): 1d array of shape
            (n_state_choices_current_period,).
        - current_period_sum_state_choices_to_state (np.ndarray): 2d array of shape
            (n_states_current_period, n_state_choices_current_period) containing the
            mapping from state choices to states for the current period.
        - resources_current_period (np.ndarray): 3d array of shape
            (n_state_choices_current_period, n_exog_savings, n_stochastic_quad_points)
             containing the resources at the beginning of the current period.
        - state_choices_current_period (np.ndarray): 1d array of shape
            (n_state_choices_current_period,) containing the state choices for the
            current period.
        - state_times_state_choice_mat_period (np.ndarray): 2d array of shape
            (n_states_current_period, n_state_choices_current_period) containing the
            mapping from states to state choices for the current period.

    """

    idx_states_current_period = np.where(state_space[:, 0] == period)[0]
    idx_state_choices_current_period = np.where(state_choice_space[:, 0] == period)[0]
    current_period_sum_state_choices_to_state = sum_state_choices_to_state[
        idx_states_current_period, :
    ][:, idx_state_choices_current_period]

    map_current_period_state_choices_to_state = map_state_choice_to_state[
        idx_state_choices_current_period
    ]
    resources_current_period = resources_beginning_of_period[
        map_current_period_state_choices_to_state
    ]
    state_choices_current_period = state_choice_space[idx_state_choices_current_period]
    state_times_state_choice_mat_period = state_times_state_choice_mat[
        idx_states_current_period, :
    ]

    return (
        idx_states_current_period,
        idx_state_choices_current_period,
        current_period_sum_state_choices_to_state,
        resources_current_period,
        state_choices_current_period,
        state_times_state_choice_mat_period,
    )
