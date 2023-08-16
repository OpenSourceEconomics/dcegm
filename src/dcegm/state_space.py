"""Functions for creating internal state space objects."""
import jax.numpy as jnp
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
        state_space (np.ndarray): 2d array of shape (n_states, n_state_vars + 1)
            which serves as a collection of all possible states. By convention,
            the first column must contain the period and the last column the
            exogenous state. Any other state variables are in between.
            E.g. if the two state variables are period and lagged choice and all choices
            are admissible in each period, the shape of the state space array is
            (n_periods * n_choices, 3).
        state_choice_space (np.ndarray): 2d array of shape
            (n_feasible_states, n_state_vars + 2) storing all feasible
            state-choice combinations. The second to last column contains the exogenous
            state. The last column includes the choice to be made at the end of
            the period (which is not a state variable).
        map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
            The shape of this object is quite complicated. For each state variable it
            has the number of possible states as rows, i.e.
            (n_poss_states_state_var_1, n_poss_states_state_var_2, ....).

    Returns:
        np.ndarray: 2d array of shape
            (n_feasible_state_choice_combs, n_choices * n_exog_processes)
            containing indices of all child nodes the agent can reach
            from a given state.

    """
    # Exogenous processes are always on the last entry of the state space. Moreover, we
    # treat all of them as admissible in each period. If there exists an absorbing
    # state, this is reflected by a 0 percent transition probability.
    n_periods, _n_choices, n_exog_processes = map_state_to_index.shape
    n_feasible_state_choice_combs = state_choice_space.shape[0]

    n_states_over_periods = state_space.shape[0] // n_periods

    map_state_to_feasible_child_nodes = np.empty(
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

                map_state_to_feasible_child_nodes[idx, exog_process] = (
                    map_state_to_index[tuple(state_vec_next)]
                    - (period + 1) * n_states_over_periods
                )

    return map_state_to_feasible_child_nodes


def create_state_choice_space(
    state_space, map_state_to_state_space_index, get_state_specific_choice_set
):
    """Create state choice space of all feasible state-choice combinations.

    Also conditional on any realization of exogenous processes.

    Args:
        state_space (np.ndarray): 2d array of shape (n_states, n_state_vars + 1)
            which serves as a collection of all possible states. By convention,
            the first column must contain the period and the last column the
            exogenous state. Any other state variables are in between.
            E.g. if the two state variables are period and lagged choice and all choices
            are admissible in each period, the shape of the state space array is
            (n_periods * n_choices, 3).
        map_state_to_state_space_index (np.ndarray): Indexer array that maps states to
            the respective index positions in the state space.
            The shape of this object is quite complicated. For each state variable it
            has the number of possible states as rows, i.e.
            (n_poss_states_state_var_1, n_poss_states_state_var_2, ....).
        get_state_specific_choice_set (Callable): User-supplied function that returns
            the set of feasible choices for a given state.

    Returns:
        tuple:

        - state_choice_space(np.ndarray): 2d array of shape
            (n_feasible_state_choice_combs, n_state_and_exog_variables + 1) containing
            the space of all feasible state-choice combinations. By convention,
            the second to last column contains the exogenous process.
            The last column always contains the choice to be made at the end of the
            period (which is not a state variable).
        - map_state_choice_vec_to_parent_state (np.ndarray): 1d array of shape
            (n_states * n_feasible_choices,) that maps from any vector of state-choice
            combinations to the respective parent state.
        - reshape_state_choice_vec_to_mat (np.ndarray): 2d array of shape
            (n_states, n_feasible_choices). For each parent state, this array can be
            used to reshape the vector of feasible state-choice combinations
            to a matrix of lagged and current choice combinations of
            shape (n_choices, n_choices).
        - transform_between_state_and_state_choice_space (jnp.ndarray): 2d boolean
            array of shape (n_states, n_states * n_feasible_choices) indicating which
            state belongs to which state-choice combination in the entire state and
            state choice space. The array is used to
            (i) contract state-choice level arrays to the state level by summing
                over state-choice combinations.
            (ii) to expand state level arrays to the state-choice level.

    """
    n_states, n_state_and_exog_variables = state_space.shape
    _n_periods, n_choices, _n_exog_processes = map_state_to_state_space_index.shape

    state_choice_space = np.zeros(
        (n_states * n_choices, n_state_and_exog_variables + 1),
        dtype=int,
    )

    map_state_choice_vec_to_parent_state = np.zeros((n_states * n_choices), dtype=int)
    reshape_state_choice_vec_to_mat = np.zeros((n_states, n_choices), dtype=int)
    transform_between_state_and_state_choice_space = np.full(
        (n_states, n_states * n_choices), fill_value=False, dtype=bool
    )

    # Ensure that states are ordered.
    period = state_space[0, 0]

    idx = 0
    idx_min = -1

    for state_idx in range(n_states):
        state_vec = state_space[state_idx]

        if period == state_vec[0]:
            idx_min = idx
            period += 1

        feasible_choice_set = get_state_specific_choice_set(
            state_vec, state_space, map_state_to_state_space_index
        )

        for choice in feasible_choice_set:
            state_choice_space[idx, :-1] = state_vec
            state_choice_space[idx, -1] = choice

            map_state_choice_vec_to_parent_state[idx] = state_idx
            reshape_state_choice_vec_to_mat[state_idx, choice] = idx - idx_min
            transform_between_state_and_state_choice_space[state_idx, idx] = True

            idx += 1

        # Fill up matrix with some state_choice index from the state, as we only use
        # this matrix to get the maximum across state_choice values and two times the
        # same value doesn't change the maximum.
        # Only used in aggregation function.
        for choice in range(n_choices):
            if choice not in feasible_choice_set:
                reshape_state_choice_vec_to_mat[state_idx, choice] = idx - idx_min - 1

    return (
        state_choice_space[:idx],
        map_state_choice_vec_to_parent_state[:idx],
        reshape_state_choice_vec_to_mat,
        transform_between_state_and_state_choice_space[:, :idx],
    )


def create_current_state_and_state_choice_objects(
    period,
    state_choices_per_period,
    states_per_period,
    state_choice_space,
    resources_beginning_of_period,
    map_state_choice_vec_to_parent_state,
    reshape_state_choice_vec_to_mat,
    transform_between_state_and_state_choice_space,
):
    """Create state and state-choice objects for the current period.

    Args:
        period (int): Current period.
        state_space (np.ndarray): 2d array of shape (n_states, n_state_variables + 1)
            containing the state space.
        state_choice_space (np.ndarray): 2d array of shape
            (n_feasible_state_choice_combs, n_states + 1) containing the space of all
            feasible state-choice combinations.
        resources_beginning_of_period (np.ndarray): 3d array of shape
            (n_states, n_exog_savings, n_stochastic_quad_points) containing the
            resourcesat the beginning of the current period for each state and
            stochastic income shock.
        map_state_choice_vec_to_parent_state (np.ndarray): 1d array of shape
            (n_states * n_feasible_choices,) that maps from any vector of state-choice
            combinations to the respective parent state.
        reshape_state_choice_vec_to_mat (np.ndarray): 2d array of shape
            (n_states, n_feasible_choices). For each parent state, this array can be
            used to reshape the vector of feasible state-choice combinations
            to a matrix of lagged and current choice combinations of
            shape (n_choices, n_choices).
        transform_between_state_and_state_choice_space (jnp.ndarray): 2d boolean
            array of shape (n_states, n_states * n_feasible_choices) indicating which
            state belongs to which state-choice combination in the entire state space
            and state-choice space. The array is used to
            (i) contract state-choice level arrays to the state level by summing
                over state-choice combinations.
            (ii) to expand state level arrays to the state-choice level.

    Returns:
        tuple:

        - idxs_state_choice_combs (np.ndarray): 1d array of shape
            (n_state_choice_combs_current,).
        - resources_current_period (np.ndarray): 3d array of shape
            (n_state_choice_combs_current, n_exog_savings, n_stochastic_quad_points)
            containing the resources at the beginning of the current period.
        - reshape_current_state_choice_vec_to_mat (np.ndarray): 2d array of shape
            (n_states_current, n_choices_current) that reshapes the current period
            vector of feasible state-choice combinations to a matrix of shape
            (n_choices, n_choices).
        - transform_between_state_and_state_choice_vec (np.ndarray): 2d boolean
            array of shape (n_states_current, n_feasible_state_choice_combs_current)
            indicating which state vector belongs to which state-choice combination in
            the current period.

    """

    _idxs_parent_states = states_per_period[period]
    idxs_state_choice_combs = state_choices_per_period[period]

    state_choice_combs = jnp.take(state_choice_space, idxs_state_choice_combs, axis=0)

    resources_current_period = jnp.take(
        resources_beginning_of_period,
        jnp.take(map_state_choice_vec_to_parent_state, idxs_state_choice_combs, axis=0),
        axis=0,
    )

    reshape_current_state_choice_vec_to_mat = jnp.take(
        reshape_state_choice_vec_to_mat, _idxs_parent_states, axis=0
    )

    transform_between_state_and_state_choice_vec = jnp.take(
        jnp.take(
            transform_between_state_and_state_choice_space, _idxs_parent_states, axis=0
        ),
        idxs_state_choice_combs,
        axis=1,
    )

    return (
        idxs_state_choice_combs,
        state_choice_combs,
        resources_current_period,
        reshape_current_state_choice_vec_to_mat,
        transform_between_state_and_state_choice_vec,
    )


def determine_states_and_state_choices_per_period(
    state_space, state_choice_space, num_periods
):
    states_per_period = {}
    state_choices_per_period = {}
    for period in range(num_periods):
        states_per_period[period] = jnp.where(state_space[:, 0] == period)[0]
        state_choices_per_period[period] = jnp.where(
            state_choice_space[:, 0] == period
        )[0]

    return state_choices_per_period, states_per_period
