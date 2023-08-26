"""Functions for creating internal state space objects."""
from typing import Dict

import jax.numpy as jnp
import numpy as np


def get_map_from_state_to_child_nodes(
    options: Dict[str, int],
    state_space: np.ndarray,
    state_choice_space: np.ndarray,
    map_state_to_index: np.ndarray,
) -> np.ndarray:
    """Create indexer array that maps states to state-specific child nodes.

    Will be a user defined function later.

    ToDo: We need to think about how to incorporate updating from state variables,
    e.g. experience.

    Args:
        options (dict): Options dictionary.
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
    n_periods = options["n_periods"]
    n_exog_states = options["n_exog_states"]

    n_feasible_state_choice_combs = state_choice_space.shape[0]

    map_state_to_feasible_child_nodes = np.empty(
        (n_feasible_state_choice_combs, n_exog_states),
        dtype=int,
    )

    current_period = 0
    first_state_index_in_period = 0
    # Loop over all state and choices by looping over the state-choice-space.
    for idx in range(n_feasible_state_choice_combs):
        state_choice_vec = state_choice_space[idx]
        period = state_choice_vec[0]

        if period < n_periods - 1:
            state_vec_next = update_endog_state_by_state_and_choice(
                state=state_choice_vec[:-1],
                choice=state_choice_vec[-1],
            )

            for exog_process in range(n_exog_states):
                state_vec_next[-1] = exog_process
                if period == current_period:
                    current_period += 1
                    first_state_index_in_period = map_state_to_index[
                        tuple(state_vec_next)
                    ]

                # We want the index every period to start at 0.
                map_state_to_feasible_child_nodes[idx, exog_process] = (
                    map_state_to_index[tuple(state_vec_next)]
                    - first_state_index_in_period
                )

    return map_state_to_feasible_child_nodes


def update_endog_state_by_state_and_choice(state, choice):
    """Get endogenous state by state and choice.

    Args:
        state (np.ndarray): 1d array of shape (n_state_vars,) containing the state.
        choice (int): Choice to be made at the end of the period.

    Returns:
        np.ndarray: 1d array of shape (n_state_vars,) containing the state of next
            period, where the endogenous part of the state is updated.

    """
    state_next = state.copy()
    state_next[0] += 1
    state_next[1] = choice
    return state_next


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


def create_period_state_and_state_choice_objects(
    state_space,
    state_choice_space,
    map_state_choice_vec_to_parent_state,
    reshape_state_choice_vec_to_mat,
    transform_between_state_and_state_choice_space,
    n_periods,
):
    """Create dictionary of state and state-choice objects for each period.

    Args:
        state_space (np.ndarray): 2d array of shape (n_states, n_state_variables)
            containing the state space.
        state_choice_space (np.ndarray): 2d array of shape
            (n_feasible_state_choice_combs, n_states + 1) containing the space of all
            feasible state-choice combinations.
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
        n_periods (int): Number of periods.

    Returns:
        dict of jnp.ndarray: Dictionary containing period-specific state and
            state-choice objects.

    """
    out = {}

    for period in range(n_periods):
        period_dict = {}
        idxs_states = jnp.where(state_space[:, 0] == period)[0]

        idxs_state_choices = jnp.where(state_choice_space[:, 0] == period)[0]
        period_dict["idxs_state_choices"] = idxs_state_choices
        period_dict["state_choice_mat"] = jnp.take(
            state_choice_space, idxs_state_choices, axis=0
        )

        period_dict["idx_state_of_state_choice"] = jnp.take(
            map_state_choice_vec_to_parent_state, idxs_state_choices, axis=0
        )

        period_dict["reshape_state_choice_vec_to_mat"] = jnp.take(
            reshape_state_choice_vec_to_mat, idxs_states, axis=0
        )

        period_dict["transform_between_state_and_state_choice_vec"] = jnp.take(
            jnp.take(
                transform_between_state_and_state_choice_space, idxs_states, axis=0
            ),
            idxs_state_choices,
            axis=1,
        )

        out[period] = period_dict

    return out
