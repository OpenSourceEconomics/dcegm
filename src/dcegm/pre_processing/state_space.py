"""Functions for creating internal state space objects."""
from typing import Callable
from typing import Dict

import numpy as np
import pandas as pd
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options


def create_state_space_and_choice_objects(
    options,
    get_state_specific_choice_set,
    get_next_period_state,
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
        dict of np.ndarray: Dictionary containing period-specific
            state and state-choice objects, with the following keys:
            - "state_choice_mat" (np.ndarray)
            - "idx_state_of_state_choice" (np.ndarray)
            - "reshape_state_choice_vec_to_mat" (callable)
            - "transform_between_state_and_state_choice_vec" (callable)

    """
    (
        state_space,
        map_state_to_state_space_index,
        states_names_without_exog,
        exog_state_names,
        n_exog_states,
        exog_state_space,
    ) = create_state_space(options)
    state_space_options = options["state_space"]
    state_space_names = states_names_without_exog + exog_state_names

    (
        state_choice_space,
        map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat,
        map_state_choice_to_index,
    ) = create_state_choice_space(
        state_space_options=state_space_options,
        state_space=state_space,
        state_space_names=state_space_names,
        map_state_to_state_space_index=map_state_to_state_space_index,
        get_state_specific_choice_set=get_state_specific_choice_set,
    )

    n_periods = state_space_options["n_periods"]

    out, batch_info = create_map_from_state_to_child_nodes(
        n_exog_states=n_exog_states,
        exog_state_space=exog_state_space,
        options=state_space_options,
        state_choice_space=state_choice_space,
        map_state_choice_to_index=map_state_choice_to_index,
        state_space=state_space,
        map_state_to_index=map_state_to_state_space_index,
        states_names_without_exog=states_names_without_exog,
        get_next_period_state=get_next_period_state,
    )

    idx_state_choice_last_period = np.where(state_choice_space[:, 0] == n_periods - 1)[
        0
    ]

    batch_info["idx_state_choice_last_period"] = idx_state_choice_last_period
    batch_info["state_idx_of_state_choice"] = map_state_choice_vec_to_parent_state[
        batch_info["batches"]
    ]
    batch_info["state_choice_mat_badge"] = {
        key: state_choice_space[:, i][batch_info["batches"]]
        for i, key in enumerate(state_space_names + ["choice"])
    }
    if not batch_info["batches_cover_all"]:
        batch_info["state_choice_mat_last_badge"] = {
            key: state_choice_space[:, i][batch_info["last_batch"]]
            for i, key in enumerate(state_space_names + ["choice"])
        }
        batch_info["last_state_idx_of_state_choice"] = (
            map_state_choice_vec_to_parent_state
        )[batch_info["last_batch"]]

    return (
        state_space,
        state_space_names,
        map_state_choice_to_index,
        exog_state_space,
        exog_state_names,
        batch_info,
    )


def create_state_space(options):
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
    state_space_options = options["state_space"]
    model_params = options["model_params"]

    n_periods = state_space_options["n_periods"]
    n_choices = len(state_space_options["choices"])

    (
        add_endog_state_func,
        endog_states_names,
        num_states_of_all_endog_states,
        num_endog_states,
        sparsity_func,
    ) = process_endog_state_specifications(
        state_space_options=state_space_options, model_params=model_params
    )

    (
        exog_states_names,
        num_states_of_all_exog_states,
        n_exog_states,
        exog_state_space,
    ) = process_exog_model_specifications(state_space_options=state_space_options)
    states_names_without_exog = ["period", "lagged_choice"] + endog_states_names

    state_space_wo_exog_list = []

    for period in range(n_periods):
        for endog_state_id in range(num_endog_states):
            for lagged_choice in range(n_choices):
                # Select the endogenous state combination
                endog_states = add_endog_state_func(endog_state_id)

                # Create the state vector without the exogenous processes
                state_without_exog = [period, lagged_choice] + endog_states

                # Transform to dictionary to call sparsity function from user
                state_dict_without_exog = {
                    states_names_without_exog[i]: state_value
                    for i, state_value in enumerate(state_without_exog)
                }

                # Check if the state is valid by calling the sparsity function
                is_state_valid = sparsity_func(**state_dict_without_exog)
                if not is_state_valid:
                    continue
                else:
                    state_space_wo_exog_list += [state_without_exog]

    state_space_wo_exog = np.array(state_space_wo_exog_list)
    state_space_wo_exog_full = np.repeat(state_space_wo_exog, n_exog_states, axis=0)
    exog_state_space_full = np.tile(exog_state_space, (state_space_wo_exog.shape[0], 1))
    state_space = np.concatenate(
        (state_space_wo_exog_full, exog_state_space_full), axis=1
    )

    # Create indexer array that maps states to indexes
    max_states = np.max(state_space, axis=0)
    map_state_to_index = np.full(max_states + 1, fill_value=-9999, dtype=int)
    state_space_tuple = tuple(state_space[:, i] for i in range(state_space.shape[1]))
    map_state_to_index[state_space_tuple] = np.arange(state_space.shape[0], dtype=int)

    return (
        state_space,
        map_state_to_index,
        states_names_without_exog,
        exog_states_names,
        n_exog_states,
        exog_state_space,
    )


def process_exog_model_specifications(state_space_options):
    if "exogenous_processes" in state_space_options:
        exog_state_names = list(state_space_options["exogenous_processes"].keys())
        dict_of_only_states = {
            key: state_space_options["exogenous_processes"][key]["states"]
            for key in exog_state_names
        }

        (
            exog_state_space,
            num_states_of_all_exog_states,
        ) = span_subspace_and_read_information(
            subdict_of_space=dict_of_only_states,
            states_names=exog_state_names,
        )
        n_exog_states = exog_state_space.shape[0]

    else:
        exog_state_names = ["dummy_exog"]
        num_states_of_all_exog_states = [1]
        n_exog_states = 1

        exog_state_space = np.array([[0]], dtype=np.int16)

    return (
        exog_state_names,
        num_states_of_all_exog_states,
        n_exog_states,
        exog_state_space,
    )


def span_subspace_and_read_information(subdict_of_space, states_names):
    all_states_values = []

    num_states_of_all_states = []
    for state_name in states_names:
        state_values = subdict_of_space[state_name]
        # Add if size_endog_state is 1, then raise Error
        num_states = len(state_values)
        num_states_of_all_states += [num_states]
        all_states_values += [state_values]

    sub_state_space = np.array(
        np.meshgrid(*all_states_values, indexing="xy")
    ).T.reshape(-1, len(states_names))

    return sub_state_space, num_states_of_all_states


def process_endog_state_specifications(state_space_options, model_params):
    """Create endog state space, to loop over in the main create state space
    function."""

    if "endogenous_states" in state_space_options:
        endog_state_keys = state_space_options["endogenous_states"].keys()
        if "sparsity_condition" in state_space_options["endogenous_states"].keys():
            endog_states_names = list(set(endog_state_keys) - {"sparsity_condition"})
            sparsity_cond_specified = True
        else:
            sparsity_cond_specified = False
            endog_states_names = list(endog_state_keys)

        (
            endog_state_space,
            num_states_of_all_endog_states,
        ) = span_subspace_and_read_information(
            subdict_of_space=state_space_options["endogenous_states"],
            states_names=endog_states_names,
        )
        num_endog_states = endog_state_space.shape[0]

    else:
        endog_states_names = []
        num_states_of_all_endog_states = []
        num_endog_states = 1

        endog_state_space = None
        sparsity_cond_specified = False

    sparsity_func = select_sparsity_function(
        sparsity_cond_specified=sparsity_cond_specified,
        state_space_options=state_space_options,
        model_params=model_params,
    )

    endog_states_add_func = create_endog_state_add_function(endog_state_space)

    return (
        endog_states_add_func,
        endog_states_names,
        num_states_of_all_endog_states,
        num_endog_states,
        sparsity_func,
    )


def select_sparsity_function(
    sparsity_cond_specified, state_space_options, model_params
):
    if sparsity_cond_specified:
        sparsity_func = determine_function_arguments_and_partial_options(
            func=state_space_options["endogenous_states"]["sparsity_condition"],
            options=model_params,
        )
    else:

        def sparsity_func(**kwargs):
            return True

    return sparsity_func


def create_endog_state_add_function(endog_state_space):
    if endog_state_space is None:

        def add_endog_states(id_endog_state):
            return []

    else:

        def add_endog_states(id_endog_state):
            return list(endog_state_space[id_endog_state])

    return add_endog_states


def create_state_choice_space(
    state_space_options,
    state_space,
    state_space_names,
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
        - transform_between_state_and_state_choice_space (np.ndarray): 2d boolean
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
    map_state_choice_to_index = np.full(
        shape=(map_state_to_state_space_index.shape + (n_choices,)),
        fill_value=-n_states * n_choices,
        dtype=int,
    )

    idx = 0
    for period in range(n_periods):
        period_states = state_space[state_space[:, 0] == period]

        period_idx = 0
        out_of_bounds_index = period_states.shape[0] * n_choices
        for state_vec in period_states:
            state_idx = map_state_to_state_space_index[tuple(state_vec)]

            state_dict = {key: state_vec[i] for i, key in enumerate(state_space_names)}
            feasible_choice_set = get_state_specific_choice_set(
                **state_dict,
            )

            for choice in range(n_choices):
                if choice in feasible_choice_set:
                    state_choice_space[idx, :-1] = state_vec
                    state_choice_space[idx, -1] = choice

                    map_state_choice_vec_to_parent_state[idx] = state_idx
                    reshape_state_choice_vec_to_mat[state_idx, choice] = period_idx
                    map_state_choice_to_index[tuple(state_vec) + (choice,)] = idx

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
        map_state_choice_to_index,
    )


def create_map_from_state_to_child_nodes(
    n_exog_states: int,
    exog_state_space: np.ndarray,
    options: Dict[str, int],
    period_specific_state_objects: Dict,
    state_choice_space: np.ndarray,
    map_state_to_index: np.ndarray,
    state_space,
    map_state_choice_to_index: np.ndarray,
    states_names_without_exog: list,
    get_next_period_state: Callable,
):
    """Create indexer array that maps states to state-specific child nodes.

    Will be a user defined function later.

    ToDo: We need to think about how to incorporate updating from state variables,
    e.g. experience.

    Args:
        options (dict): Options dictionary.
        period_specific_state_objects (np.ndarray): Dictionary containing
            period-specific state and state-choice objects, with the following keys:
            - "state_choice_mat" (np.ndarray)
            - "idx_state_of_state_choice" (np.ndarray)
            - "reshape_state_choice_vec_to_mat" (callable)
            - "transform_between_state_and_state_choice_vec" (callable)
        map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
            The shape of this object is quite complicated. For each state variable it
            has the number of potential states as rows, i.e.
            (n_potential_states_state_var_1, n_potential_states_state_var_2, ....).
        get_next_period_state (Callable): User-supplied function that
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

    n_exog_vars = exog_state_space.shape[1]

    map_state_choice_to_feasible_child_states = np.full(
        (state_choice_space.shape[0], n_exog_states), fill_value=-9999, dtype=int
    )

    exog_states_tuple = tuple(exog_state_space[:, i] for i in range(n_exog_vars))
    current_state_choice_idx = -1
    for period in range(n_periods - 1):
        end_of_prev_period_index = current_state_choice_idx + 1
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
            current_state_choice_idx = end_of_prev_period_index + idx
            current_state = state_choice_vec[:-1]
            current_state_without_exog = current_state[:-n_exog_vars]

            # The update function is not allowed to depend on exogenous process. It is
            # only about the exogenous part.
            state_dict_without_exog = {
                key: current_state_without_exog[i]
                for i, key in enumerate(states_names_without_exog)
            }

            endog_state_update = get_next_period_state(
                **state_dict_without_exog, choice=state_choice_vec[-1]
            )

            state_dict_without_exog.update(endog_state_update)

            states_next_tuple = (
                tuple(
                    np.full(
                        n_exog_states,
                        fill_value=state_dict_without_exog[key],
                        dtype=int,
                    )
                    for key in states_names_without_exog
                )
                + exog_states_tuple
            )

            child_idxs = map_state_to_index[states_next_tuple]
            map_state_to_feasible_child_nodes_period[idx, :] = (
                child_idxs - idx_min_state_space_next_period
            )
            map_state_choice_to_feasible_child_states[
                current_state_choice_idx, :
            ] = child_idxs

            period_specific_state_objects[period][
                "idx_feasible_child_nodes"
            ] = np.array(map_state_to_feasible_child_nodes_period, dtype=int)

    (
        batches_list,
        unique_child_state_choice_idxs_list,
        state_choice_times_exog_child_state_idxs_list,
    ) = determine_optimal_batch_size(
        state_choice_space,
        n_periods,
        map_state_choice_to_feasible_child_states,
        map_state_choice_to_index,
        state_space,
    )
    if len(batches_list) == 1:
        # This is the case of a two period model. Then by construction there is only one
        # batch which covers the first period.
        batches_cover_all = True
    else:
        # In the case of more periods we determine if the last two batches have equal
        # size
        batches_cover_all = len(batches_list[-1]) != len(batches_list[-2])

    if not batches_cover_all:
        batch_array = np.array(batches_list[:-1])
        state_choice_times_exog_child_state_idxs = np.array(
            state_choice_times_exog_child_state_idxs_list[:-1]
        )

        # There can be also be an uneven number of child states across batches. The
        # indexes recorded in state_choice_times_exog_child_state_idxs only contain
        # the indexes up the length. So we can just fill up without of bounds indexes.
        # We also test this here
        max_n_state_unique_in_batches = list(
            map(lambda x: x.shape[0], unique_child_state_choice_idxs_list[:-1])
        )

        if not np.all(
            np.equal(
                np.array(max_n_state_unique_in_batches) - 1,
                np.max(state_choice_times_exog_child_state_idxs, axis=(1, 2)),
            )
        ):
            raise ValueError(
                "\n\nInternal error in the batch creation \n\n. "
                "Please contact developer."
            )

        n_batches = batch_array.shape[0]
        n_choices = unique_child_state_choice_idxs_list[0].shape[1]
        max_n_state_accross_batches = np.max(max_n_state_unique_in_batches)
        unique_child_state_choice_idxs = np.full(
            (n_batches, max_n_state_accross_batches, n_choices),
            fill_value=-9999,
            dtype=int,
        )

        for id_batch in range(n_batches):
            unique_child_state_choice_idxs[
                id_batch, : max_n_state_unique_in_batches[id_batch], :
            ] = unique_child_state_choice_idxs_list[id_batch]

        additional_information = {
            "last_batch": batches_list[-1],
            "last_unique_child_state_choice_idxs": unique_child_state_choice_idxs_list[
                -1
            ],
            "last_state_choice_times_exog_child_state_idxs": state_choice_times_exog_child_state_idxs_list[
                -1
            ],
        }
    else:
        batch_array = np.array(batches_list)
        unique_child_state_choice_idxs = np.array(unique_child_state_choice_idxs_list)
        state_choice_times_exog_child_state_idxs = np.array(
            state_choice_times_exog_child_state_idxs_list
        )
        additional_information = {}

    batches_information = {
        **additional_information,
        "batches_cover_all": batches_cover_all,
        "batches": batch_array,
        "unique_child_state_choice_idxs": unique_child_state_choice_idxs,
        "child_state_to_state_choice_exog": state_choice_times_exog_child_state_idxs,
        "n_state_choices": state_choice_space.shape[0],
    }

    return period_specific_state_objects, batches_information


def determine_optimal_batch_size(
    state_choice_space,
    n_periods,
    map_state_choice_to_feasible_child_states,
    map_state_choice_to_index,
    state_space,
):
    state_choice_space_wo_last = state_choice_space[
        state_choice_space[:, 0] < n_periods - 1
    ]
    state_choice_index_back = np.arange(state_choice_space_wo_last.shape[0], dtype=int)

    # Filter out last period state_choice_ids
    child_states_idx_backward = map_state_choice_to_feasible_child_states[
        state_choice_space[:, 0] < n_periods - 1
    ]
    child_states = np.take(state_space, child_states_idx_backward, axis=0)
    n_state_vars = state_space.shape[1]

    size_last_batch = state_choice_space[
        state_choice_space[:, 0] == state_choice_space_wo_last[-1, 0]
    ].shape[0]

    batch_not_found = True
    current_batch_size = size_last_batch
    need_to_reduce_batchsize = False
    while batch_not_found:
        if need_to_reduce_batchsize:
            current_batch_size = int(current_batch_size * 0.95)
            need_to_reduce_batchsize = False
        # Split state choice indexes in
        index_to_spilt = np.arange(
            current_batch_size,
            state_choice_index_back.shape[0],
            current_batch_size,
        )

        batches_to_check = np.split(
            np.flip(state_choice_index_back),
            index_to_spilt,
        )
        child_state_to_state_choice_times_exog = []
        unique_child_state_choice_idxs = []

        for i, batch in enumerate(batches_to_check):
            child_states_idxs = map_state_choice_to_feasible_child_states[batch]
            unique_child_states, unique_ids, inverse_ids = np.unique(
                child_states_idxs, return_index=True, return_inverse=True
            )

            child_state_to_state_choice_times_exog += [
                inverse_ids.reshape(child_states_idxs.shape)
            ]

            # Get child states for current batch of state choices
            child_states_batch = np.take(child_states, batch, axis=0).reshape(
                -1, n_state_vars
            )

            # Make tuple out of columns of child states
            child_states_tuple = tuple(
                child_states_batch[:, i] for i in range(n_state_vars)
            )

            # Get ids of state choices for each child state
            state_choice_idxs_childs = map_state_choice_to_index[child_states_tuple]
            # Save the unique child states
            unique_child_state_choice_idxs += [state_choice_idxs_childs[unique_ids]]

            # Get minimum of the positive numbers in state_choice_idxs_childs
            min_state_choice_idx = np.min(
                state_choice_idxs_childs[state_choice_idxs_childs > 0]
            )
            # Now check if the smallest index of the child state choices is larger than
            # the maximum index of the batch, i.e. if all state choice relevant to
            # solve the current state choices of the batch are in previous batches
            if batch.max() > min_state_choice_idx:
                batch_not_found = True
                need_to_reduce_batchsize = True
                break

        if not need_to_reduce_batchsize:
            batch_not_found = False

        print("The batch size of the backwards induction is ", current_batch_size)
        print("It failed in batch ", i)

    return (
        batches_to_check,
        unique_child_state_choice_idxs,
        child_state_to_state_choice_times_exog,
    )


def inspect_state_space(
    options: Dict[str, float],
):
    """Creates a data frame of all potential states and a feasibility flag."""
    state_space_options = options["state_space"]
    model_params = options["model_params"]

    n_periods = state_space_options["n_periods"]
    n_choices = len(state_space_options["choices"])

    (
        add_endog_state_func,
        endog_states_names,
        _,
        num_endog_states,
        sparsity_func,
    ) = process_endog_state_specifications(
        state_space_options=state_space_options, model_params=model_params
    )

    (
        exog_states_names,
        _,
        n_exog_states,
        exog_state_space,
    ) = process_exog_model_specifications(state_space_options=state_space_options)

    states_names_without_exog = ["period", "lagged_choice"] + endog_states_names

    state_space_wo_exog_list = []
    is_feasible_list = []

    for period in range(n_periods):
        for endog_state_id in range(num_endog_states):
            for lagged_choice in range(n_choices):
                # Select the endogenous state combination
                endog_states = add_endog_state_func(endog_state_id)

                # Create the state vector without the exogenous processes
                state_without_exog = [period, lagged_choice] + endog_states
                state_space_wo_exog_list += [state_without_exog]

                # Transform to dictionary to call sparsity function from user
                state_dict_without_exog = {
                    states_names_without_exog[i]: state_value
                    for i, state_value in enumerate(state_without_exog)
                }

                is_state_feasible = sparsity_func(**state_dict_without_exog)
                is_feasible_list += [is_state_feasible]

    state_space_wo_exog = np.array(state_space_wo_exog_list)
    state_space_wo_exog_full = np.repeat(state_space_wo_exog, n_exog_states, axis=0)
    exog_state_space_full = np.tile(exog_state_space, (state_space_wo_exog.shape[0], 1))

    state_space = np.concatenate(
        (state_space_wo_exog_full, exog_state_space_full), axis=1
    )

    state_space_df = pd.DataFrame(
        state_space, columns=states_names_without_exog + exog_states_names
    )
    is_feasible_array = np.array(is_feasible_list, dtype=bool)

    state_space_df["is_feasible"] = np.repeat(is_feasible_array, n_exog_states, axis=0)

    return state_space_df
