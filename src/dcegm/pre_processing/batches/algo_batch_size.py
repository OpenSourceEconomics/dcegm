import numpy as np


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

        for i, batch in enumerate(batches_to_check):
            # First get all child states and a mapping from the state-choice to the
            # different child states due to exogenous change of states.
            child_states_idxs = map_state_choice_to_child_states[batch]
            unique_child_states, inverse_ids = np.unique(
                child_states_idxs, return_index=False, return_inverse=True
            )
            child_states_to_integrate_exog += [
                inverse_ids.reshape(child_states_idxs.shape)
            ]

            # Next we use the child state indexes to get all unique child states and
            # their corresponding state-choices.
            child_states_batch = np.take(state_space, unique_child_states, axis=0)
            child_states_tuple = tuple(
                child_states_batch[:, i] for i in range(n_state_vars)
            )
            unique_state_choice_idxs_childs = map_state_choice_to_index[
                child_states_tuple
            ]

            # Now we create a mapping from the child-state choices back to the states
            # with state-choices in columns for the choices
            (
                unique_child_state_choice_idxs,
                inverse_child_state_choice_ids,
            ) = np.unique(
                unique_state_choice_idxs_childs, return_index=False, return_inverse=True
            )

            # Treat invalid choices:
            if unique_child_state_choice_idxs[-1] == invalid_state_idx:
                unique_child_state_choice_idxs = unique_child_state_choice_idxs[:-1]
                inverse_child_state_choice_ids[
                    inverse_child_state_choice_ids
                    >= np.max(inverse_child_state_choice_ids)
                ] = out_of_bounds_state_choice_idx

            # Save the mapping from child-state-choices to child-states
            child_state_choices_to_aggr_choice += [
                inverse_child_state_choice_ids.reshape(
                    unique_state_choice_idxs_childs.shape
                )
            ]
            # And the list of the unique child states.
            child_state_choice_idxs_to_interpolate += [unique_child_state_choice_idxs]

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
    )
