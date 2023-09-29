"""Functions for creating internal state space objects."""
from functools import reduce
from typing import Callable
from typing import Dict

import jax.numpy as jnp
import numpy as np


def create_state_space_and_choice_objects(
    state_space_options,
    get_state_specific_choice_set,
    update_endog_state_by_state_and_choice,
    exog_mapping,
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
        n_periods (int): Number of periods.

    Returns:
        dict of jnp.ndarray: Dictionary containing period-specific
            state and state-choice objects, with the following keys:
            - "state_choice_mat" (jnp.ndarray)
            - "idx_state_of_state_choice" (jnp.ndarray)
            - "reshape_state_choice_vec_to_mat" (callable)
            - "transform_between_state_and_state_choice_vec" (callable)

    """
    state_space, map_state_to_state_space_index = create_state_space(
        state_space_options
    )

    (
        state_choice_space,
        map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat,
    ) = create_state_choice_space(
        state_space_options=state_space_options,
        state_space=state_space,
        map_state_to_state_space_index=map_state_to_state_space_index,
        get_state_specific_choice_set=get_state_specific_choice_set,
    )

    n_periods = state_space_options["n_periods"]

    out = {}

    for period in range(n_periods):
        period_dict = {}
        idxs_states = jnp.where(state_space[:, 0] == period)[0]

        idxs_state_choices_period = jnp.where(state_choice_space[:, 0] == period)[0]
        state_choice_mat = jnp.take(
            state_choice_space, idxs_state_choices_period, axis=0
        )

        period_dict["state_choice_mat"] = state_choice_mat

        period_dict["idx_parent_states"] = jnp.take(
            map_state_choice_vec_to_parent_state, idxs_state_choices_period, axis=0
        )

        period_dict["reshape_state_choice_vec_to_mat"] = jnp.take(
            reshape_state_choice_vec_to_mat, idxs_states, axis=0
        )

        out[period] = period_dict

    out = create_map_from_state_to_child_nodes(
        options=state_space_options,
        period_specific_state_objects=out,
        map_state_to_index=map_state_to_state_space_index,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    for period in range(n_periods):
        out[period]["state_choice_mat"] = {
            "period": out[period]["state_choice_mat"][:, 0],
            "lagged_choice": out[period]["state_choice_mat"][:, 1],
            "married": out[period]["state_choice_mat"][:, 2],
            "ltc": exog_mapping[:, 0][out[period]["state_choice_mat"][:, 3]],
            "job_offer": exog_mapping[:, 1][out[period]["state_choice_mat"][:, 3]],
            "choice": out[period]["state_choice_mat"][:, 4],
        }

    state_space = {
        "period": state_space[:, 0],
        "lagged_choice": state_space[:, 1],
        "married": state_space[:, 2],
        "ltc": exog_mapping[:, 0][state_space[:, 3]],
        "job_offer": exog_mapping[:, 1][state_space[:, 3]],
    }

    return out, state_space


def create_state_space(state_space_options):
    """Create state space object and indexer.

    We need to add the convention for the state space objects.

    Args:
        options (dict): Options dictionary.

    Returns:
        tuple:

        - state_vars (list): List of state variables.
        - state_space (np.ndarray): 2d array of shape (n_states, n_state_variables + 1)
            which serves as a collection of all possible states. By convention,
            the first column must contain the period and the last column the
            exogenous processes. Any other state variables are in between.
            E.g. if the two state variables are period and lagged choice and all choices
            are admissible in each period, the shape of the state space array is
            (n_periods * n_choices, 3).
        - map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
            The shape of this object is quite complicated. For each state variable it
            has the number of possible states as rows, i.e.
            (n_poss_states_state_var_1, n_poss_states_state_var_2, ....).

    """
    n_periods = state_space_options["n_periods"]
    n_choices = len(state_space_options["choices"])

    (
        add_endog_state_func,
        num_of_endog_combinations,
        endog_states_names,
        num_endog_states,
    ) = determine_endog_states_and_create_add_function(state_space_options)

    state_names_without_exog = ["period", "lagged_choice"] + endog_states_names

    if "exogenous_processes" in state_space_options:
        n_exog = 0
        for exog_processes in state_space_options["exogenous_processes"].keys():
            n_exog += len(
                state_space_options["exogenous_processes"][exog_processes]["states"]
            )

    shape = [n_periods, n_choices] + num_endog_states

    map_state_to_index = np.full(shape, -9999, dtype=np.int64)
    state_space_list = []

    index = 0
    for period in range(n_periods):
        for lagged_choice in range(n_choices):
            for endog_state_id in range(num_of_endog_combinations):
                endog_states = add_endog_state_func(endog_state_id)
                state_without_exog = [period, lagged_choice] + endog_states
                {
                    state_names_without_exog[i]: state_value
                    for i, state_value in enumerate(state_without_exog)
                }
                state_space_list += [state_without_exog]
                map_state_to_index[tuple(state_without_exog)] = index
                index += 1

    np.array(state_space_list)
    # breakpoint()


def determine_endog_states_and_create_add_function(state_space_options):
    num_of_endog_combinations = 1
    if "endogenous_states" in state_space_options:
        endog_all_states_values = []
        endog_states_names = list(state_space_options["endogenous_states"].keys())
        num_endog_states = []
        for endog_state in endog_states_names:
            endog_state_values = state_space_options["endogenous_states"][endog_state]
            # Add if size_endog_state is 1, then raise Error
            num_states = len(endog_state_values)
            num_of_endog_combinations *= num_states
            num_endog_states += [num_states]
            endog_all_states_values += [endog_state_values]

        num_of_endog_state_vars = len(state_space_options["endogenous_states"].keys())
        endog_state_space = np.array(
            reduce(np.meshgrid, endog_all_states_values)
        ).T.reshape(-1, num_of_endog_state_vars)
    else:
        endog_state_space = None

    endog_states_add_func = create_endog_state_add_function(
        endog_state_space, num_of_endog_combinations
    )

    return (
        endog_states_add_func,
        num_of_endog_combinations,
        endog_states_names,
        num_endog_states,
    )


def create_endog_state_add_function(endog_state_space, num_of_endog_combinations):
    if num_of_endog_combinations == 1:

        def add_endog_states(id_endog_state):
            return []

    else:

        def add_endog_states(id_endog_state):
            return list(endog_state_space[id_endog_state])

    return add_endog_states


def create_state_choice_space(
    state_space_options,
    state_space,
    map_state_to_state_space_index,
    get_state_specific_choice_set,
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
        map_state_to_state_space_index (np.ndarray): Indexer array that maps
            a period-specific state vector to the respective index positions in the
            state space.
            The shape of this object is quite complicated. For each state variable it
            has the number of potential states as rows, i.e.
            (n_potential_states_state_var_1, n_potential_states_state_var_2, ....).
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

    n_periods = state_space_options["n_periods"]
    n_choices = len(state_space_options["choices"])

    state_choice_space = np.zeros(
        (n_states * n_choices, n_state_and_exog_variables + 1),
        dtype=int,
    )

    map_state_choice_vec_to_parent_state = np.zeros((n_states * n_choices), dtype=int)
    reshape_state_choice_vec_to_mat = np.zeros((n_states, n_choices), dtype=int)

    idx = 0
    for period in range(n_periods):
        period_states = state_space[state_space[:, 0] == period]

        period_idx = 0
        out_of_bounds_index = period_states.shape[0] * n_choices
        for state_vec in period_states:
            state_idx = map_state_to_state_space_index[tuple(state_vec)]

            feasible_choice_set = get_state_specific_choice_set(
                state=state_vec,
            )

            for choice in range(n_choices):
                if choice in feasible_choice_set:
                    state_choice_space[idx, :-1] = state_vec
                    state_choice_space[idx, -1] = choice

                    map_state_choice_vec_to_parent_state[idx] = state_idx
                    reshape_state_choice_vec_to_mat[state_idx, choice] = period_idx

                    period_idx += 1
                    idx += 1
                else:
                    reshape_state_choice_vec_to_mat[
                        state_idx, choice
                    ] = out_of_bounds_index

    return (
        state_choice_space[:idx],
        map_state_choice_vec_to_parent_state[:idx],
        reshape_state_choice_vec_to_mat,
    )


def create_map_from_state_to_child_nodes(
    options: Dict[str, int],
    period_specific_state_objects: np.ndarray,
    map_state_to_index: np.ndarray,
    update_endog_state_by_state_and_choice: Callable,
):
    """Create indexer array that maps states to state-specific child nodes.

    Will be a user defined function later.

    ToDo: We need to think about how to incorporate updating from state variables,
    e.g. experience.

    Args:
        options (dict): Options dictionary.
        period_specific_state_objects (np.ndarray): Dictionary containing
            period-specific state and state-choice objects, with the following keys:
            - "state_choice_mat" (jnp.ndarray)
            - "idx_state_of_state_choice" (jnp.ndarray)
            - "reshape_state_choice_vec_to_mat" (callable)
            - "transform_between_state_and_state_choice_vec" (callable)
        map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
            The shape of this object is quite complicated. For each state variable it
            has the number of potential states as rows, i.e.
            (n_potential_states_state_var_1, n_potential_states_state_var_2, ....).
        update_endog_state_by_state_and_choice (Callable): User-supplied function that
            updates the endogenous state variables conditional on the current state and
            choice.

    Returns:
        tuple:
            period_specific_state_space_objects (np.ndarray): 2d array of shape
                (n_feasible_state_choice_combs, n_choices * n_exog_processes)
                containing indices of all child nodes the agent can reach
                from a given state.

    """

    # Exogenous processes are always on the last entry of the state space. Moreover, we
    # treat all of them as admissible in each period. If there exists an absorbing
    # state, this is reflected by a 0 percent transition probability.
    n_periods = options["n_periods"]
    n_exog_states = 0

    for exog_process in options["exogenous_processes"].keys():
        n_exog_states += len(options["exogenous_processes"][exog_process]["states"])

    for period in range(n_periods - 1):
        period_dict = period_specific_state_objects[period]
        idx_min_state_space_next_period = map_state_to_index[
            tuple(period_specific_state_objects[period + 1]["state_choice_mat"][0, :-1])
        ]

        state_choice_space_period = period_dict["state_choice_mat"]

        map_state_to_feasible_child_nodes_period = np.empty(
            (state_choice_space_period.shape[0], n_exog_states),
            dtype=int,
        )

        # Loop over all state-choice combinations in period.
        for idx, state_choice_vec in enumerate(state_choice_space_period):
            state_vec_next = update_endog_state_by_state_and_choice(
                state=np.array(state_choice_vec[:-1]),
                choice=np.array(state_choice_vec[-1]),
                # ToDo: options
                # all optional
            )

            for exog_process in range(n_exog_states):
                state_vec_next[-1] = exog_process
                # We want the index every period to start at 0.
                map_state_to_feasible_child_nodes_period[idx, exog_process] = (
                    map_state_to_index[tuple(state_vec_next)]
                    - idx_min_state_space_next_period
                )

            period_specific_state_objects[period][
                "idx_feasible_child_nodes"
            ] = jnp.array(map_state_to_feasible_child_nodes_period, dtype=int)

    return period_specific_state_objects
