import numpy as np

from dcegm.pre_processing.batches.child_state_dedup import compute_child_dedup_for_batch


def determine_optimal_batch_size(
    bool_state_choices_to_batch,
    state_choice_space,
    map_state_choice_to_child_states,
    map_state_choice_to_index,
    state_space,
):
    # Get invalid state idx, by looking at the index mapping dtype
    invalid_state_idx = np.iinfo(map_state_choice_to_index.dtype).max
    # Get out of bound state choice idx, by taking the number of state choices + 1
    out_of_bounds_state_choice_idx = state_choice_space.shape[0] + 1

    state_choice_space_to_batch = state_choice_space[bool_state_choices_to_batch]

    child_states_of_state_choices_to_batch = map_state_choice_to_child_states[
        bool_state_choices_to_batch
    ]
    # Order by child index to solve state choices in the same child states together
    # Use first child state of the n_exog_states of each child states, because
    # rows are the same in the child states mapping array. Making this more robust
    # by selecting the minimum in each row (because of sparsity)
    min_child_states_per_state_choice = np.min(
        child_states_of_state_choices_to_batch, axis=1
    )
    sort_index_by_child_states = np.argsort(min_child_states_per_state_choice)

    idx_state_choice_raw = np.where(bool_state_choices_to_batch)[0]
    state_choice_index_back = np.take(
        idx_state_choice_raw, sort_index_by_child_states, axis=0
    )

    n_state_vars = state_space.shape[1]

    size_last_period = state_choice_space[
        state_choice_space[:, 0] == state_choice_space_to_batch[-1, 0]
    ].shape[0]

    batch_not_found = True
    current_batch_size = size_last_period
    need_to_reduce_batchsize = False

    while batch_not_found:
        if need_to_reduce_batchsize:
            current_batch_size = int(current_batch_size * 0.98)
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

        child_states_to_integrate_exog = []
        child_state_choices_to_aggr_choice = []
        child_state_choice_idxs_to_interpolate = []
        representative_parent_state_choice_for_child = []
        unique_child_states_list = []
        representative_parent_state_choice_per_child_state_list = []
        state_row_for_state_choice_list = []

        for i, batch in enumerate(batches_to_check):
            (
                child_states_to_integrate_exog_batch,
                child_state_choices_to_aggr_choice_batch,
                unique_child_state_choice_idxs,
                representative_parent_state_choice_batch,
                unique_child_states_batch,
                representative_parent_state_choice_per_child_state_batch,
                state_row_for_state_choice_batch,
            ) = compute_child_dedup_for_batch(
                batch=batch,
                map_state_choice_to_child_states=map_state_choice_to_child_states,
                map_state_choice_to_index=map_state_choice_to_index,
                state_space=state_space,
                n_state_vars=n_state_vars,
                invalid_state_idx=invalid_state_idx,
                out_of_bounds_state_choice_idx=out_of_bounds_state_choice_idx,
            )
            child_states_to_integrate_exog += [child_states_to_integrate_exog_batch]
            child_state_choices_to_aggr_choice += [
                child_state_choices_to_aggr_choice_batch
            ]
            child_state_choice_idxs_to_interpolate += [unique_child_state_choice_idxs]
            representative_parent_state_choice_for_child += [
                representative_parent_state_choice_batch
            ]
            unique_child_states_list += [unique_child_states_batch]
            representative_parent_state_choice_per_child_state_list += [
                representative_parent_state_choice_per_child_state_batch
            ]
            state_row_for_state_choice_list += [state_row_for_state_choice_batch]

            # Now check if the smallest index of the child state choices is larger than
            # the maximum index of the batch, i.e. if all state choice relevant to
            # solve the current state choices of the batch are in previous batches
            min_state_choice_idx = np.min(unique_child_state_choice_idxs)
            if batch.max() >= min_state_choice_idx:
                batch_not_found = True
                need_to_reduce_batchsize = True
                break

        print("The batch size of the backwards induction is ", current_batch_size)

        if not need_to_reduce_batchsize:
            batch_not_found = False

    return (
        batches_to_check,
        child_state_choice_idxs_to_interpolate,
        child_state_choices_to_aggr_choice,
        child_states_to_integrate_exog,
        representative_parent_state_choice_for_child,
        unique_child_states_list,
        representative_parent_state_choice_per_child_state_list,
        state_row_for_state_choice_list,
    )
