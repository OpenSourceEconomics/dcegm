import numpy as np


def create_batches_and_information(
    model_structure,
    model_funcs,
    options,
):
    n_periods = options["state_space"]["n_periods"]
    state_choice_space = model_structure["state_choice_space"]
    state_space = model_structure["state_space"]
    state_space_names = model_structure["state_space_names"]
    map_state_choice_to_parent_state = model_structure[
        "map_state_choice_to_parent_state"
    ]
    map_state_choice_to_child_states = model_structure[
        "map_state_choice_to_child_states"
    ]
    map_state_choice_to_index = model_structure["map_state_choice_to_index"]

    if n_periods == 2:
        # In the case of a two period model, we just need the information of the last
        # two periods
        batch_info = {
            "two_period_model": True,
            "n_state_choices": state_choice_space.shape[0],
        }
        batch_info = add_last_two_period_information(
            n_periods=n_periods,
            state_choice_space=state_choice_space,
            map_state_choice_to_parent_state=map_state_choice_to_parent_state,
            map_state_choice_to_child_states=map_state_choice_to_child_states,
            map_state_choice_to_index=map_state_choice_to_index,
            state_space_names=state_space_names,
            state_space=state_space,
            batch_info=batch_info,
        )

        return batch_info

    (
        batches_list,
        unique_child_state_choice_idxs_list,
        state_choice_times_exog_child_state_idxs_list,
    ) = determine_optimal_batch_size(
        state_choice_space,
        n_periods,
        map_state_choice_to_child_states,
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
        batches_cover_all = len(batches_list[-1]) == len(batches_list[-2])

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

        state_choices_last_badge = {
            key: state_choice_space[:, i][batch_array]
            for i, key in enumerate(state_space_names + ["choice"])
        }
        parent_states_idx_state_choice_last_batch = (map_state_choice_to_parent_state)[
            batch_array
        ]

        additional_information = {
            "last_batch": batches_list[-1],
            "last_unique_child_state_choice_idxs": unique_child_state_choice_idxs_list[
                -1
            ],
            "last_state_choice_times_exog_child_state_idxs": state_choice_times_exog_child_state_idxs_list[
                -1
            ],
            "state_choices_last_badge": state_choices_last_badge,
            "parent_states_idx_state_choice_last_batch": parent_states_idx_state_choice_last_batch,
        }
    else:
        batch_array = np.array(batches_list)
        unique_child_state_choice_idxs = np.array(unique_child_state_choice_idxs_list)
        state_choice_times_exog_child_state_idxs = np.array(
            state_choice_times_exog_child_state_idxs_list
        )
        additional_information = {}

    parent_state_idx_of_state_choice = map_state_choice_to_parent_state[batch_array]
    state_choices_batches = {
        key: state_choice_space[:, i][batch_array]
        for i, key in enumerate(state_space_names + ["choice"])
    }

    batch_info = {
        **additional_information,
        "batches_cover_all": batches_cover_all,
        "batches": batch_array,
        "unique_child_state_choice_idxs": unique_child_state_choice_idxs,
        "child_state_to_state_choice_exog": state_choice_times_exog_child_state_idxs,
        "n_state_choices": state_choice_space.shape[0],
        "parent_states_idx_state_choice": parent_state_idx_of_state_choice,
        "state_choices_batches": state_choices_batches,
    }
    batch_info = add_last_two_period_information(
        n_periods,
        state_choice_space,
        map_state_choice_to_parent_state,
        state_space_names,
        batch_info,
    )

    return batch_info


def add_last_two_period_information(
    n_periods,
    state_choice_space,
    map_state_choice_to_parent_state,
    map_state_choice_to_child_states,
    map_state_choice_to_index,
    state_space_names,
    state_space,
    batch_info,
):
    # Select state_choice idxs in final period
    idx_state_choice_final_period = np.where(state_choice_space[:, 0] == n_periods - 1)[
        0
    ]
    # To solve the second last period, we need the child states in the last period
    # and the corresponding matrix, where each row is a state with the state choice
    # ids as entry in each choice
    states_final_period = state_space[state_space[:, 0] == n_periods - 1]
    # Now construct a tuple for indexing
    n_state_vars = states_final_period.shape[1]
    states_tuple = tuple(states_final_period[:, i] for i in range(n_state_vars))

    # Now get the matrix we use for choice aggregation
    state_to_choices_final_period = map_state_choice_to_index[states_tuple]
    # Normalize to be able to index in the interpolated values
    min_val = np.min(state_to_choices_final_period[state_to_choices_final_period > 0])
    state_to_choices_final_period -= min_val

    idx_state_choice_second_last_period = np.where(
        state_choice_space[:, 0] == n_periods - 2
    )[0]
    # Also normalize the state choice idxs
    child_states_second_last_period = map_state_choice_to_child_states[
        idx_state_choice_second_last_period
    ]
    min_val = np.min(
        child_states_second_last_period[child_states_second_last_period > 0]
    )
    child_states_second_last_period -= min_val

    # Also add parent states in last period
    parent_states_final_period = map_state_choice_to_parent_state[
        idx_state_choice_final_period
    ]
    batch_info = {
        **batch_info,
        "idx_state_choices_final_period": idx_state_choice_final_period,
        "idx_state_choices_second_last_period": idx_state_choice_second_last_period,
        "idxs_parent_states_final_period": parent_states_final_period,
        "state_to_choices_final_period": state_to_choices_final_period,
        "child_states_second_last_period": child_states_second_last_period,
    }

    # Also add state choice mat as dictionary for each of the two periods
    for idx, period_name in [
        (idx_state_choice_final_period, "final"),
        (idx_state_choice_second_last_period, "second_last"),
    ]:
        batch_info[f"state_choice_mat_{period_name}_period"] = {
            key: state_choice_space[:, i][idx]
            for i, key in enumerate(state_space_names + ["choice"])
        }
    return batch_info


def determine_optimal_batch_size(
    state_choice_space,
    n_periods,
    map_state_choice_to_child_states,
    map_state_choice_to_index,
    state_space,
):
    state_choice_space_wo_last_two = state_choice_space[
        state_choice_space[:, 0] < n_periods - 2
    ]
    state_choice_index_back = np.arange(
        state_choice_space_wo_last_two.shape[0], dtype=int
    )

    # Filter out last period state_choice_ids
    child_states_idx_backward = map_state_choice_to_child_states[
        state_choice_space[:, 0] < n_periods - 2
    ]
    child_states = np.take(state_space, child_states_idx_backward, axis=0)
    n_state_vars = state_space.shape[1]

    size_last_batch = state_choice_space[
        state_choice_space[:, 0] == state_choice_space_wo_last_two[-1, 0]
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
            child_states_idxs = map_state_choice_to_child_states[batch]
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
