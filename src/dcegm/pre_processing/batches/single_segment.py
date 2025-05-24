import numpy as np

from dcegm.pre_processing.batches.algo_batch_size import determine_optimal_batch_size


def create_single_segment_of_batches(bool_state_choices_to_batch, model_structure):
    """Create a single segment of evenly sized batches.

    If the last batch is not evenly we correct it.

    """

    state_choice_space = model_structure["state_choice_space"]

    state_space = model_structure["state_space"]
    discrete_states_names = model_structure["discrete_states_names"]

    map_state_choice_to_parent_state = model_structure[
        "map_state_choice_to_parent_state"
    ]
    map_state_choice_to_child_states = model_structure[
        "map_state_choice_to_child_states"
    ]
    map_state_choice_to_index = model_structure["map_state_choice_to_index_with_proxy"]

    (
        batches_list,
        child_state_choice_idxs_to_interp_list,
        child_state_choices_to_aggr_choice_list,
        child_states_to_integrate_stochastic_list,
    ) = determine_optimal_batch_size(
        bool_state_choices_to_batch=bool_state_choices_to_batch,
        state_choice_space=state_choice_space,
        map_state_choice_to_child_states=map_state_choice_to_child_states,
        map_state_choice_to_index=map_state_choice_to_index,
        state_space=state_space,
    )

    (
        batches_list,
        child_states_to_integrate_stochastic_list,
        child_state_choices_to_aggr_choice_list,
        child_state_choice_idxs_to_interp_list,
        batches_cover_all,
        last_batch_info,
    ) = correct_for_uneven_last_batch(
        batches_list,
        child_states_to_integrate_stochastic_list,
        child_state_choices_to_aggr_choice_list,
        child_state_choice_idxs_to_interp_list,
        state_choice_space,
        map_state_choice_to_parent_state,
        discrete_states_names,
    )

    single_batch_segment_info = prepare_and_align_batch_arrays(
        batches_list,
        child_states_to_integrate_stochastic_list,
        child_state_choices_to_aggr_choice_list,
        child_state_choice_idxs_to_interp_list,
        state_choice_space,
        map_state_choice_to_parent_state,
        discrete_states_names,
    )
    single_batch_segment_info["batches_cover_all"] = batches_cover_all
    if not batches_cover_all:
        single_batch_segment_info["last_batch_info"] = last_batch_info

    return single_batch_segment_info


def correct_for_uneven_last_batch(
    batches_list,
    child_states_to_integrate_stochastic_list,
    child_state_choices_to_aggr_choice_list,
    child_state_choice_idxs_to_interp_list,
    state_choice_space,
    map_state_choice_to_parent_state,
    discrete_states_names,
):
    """Check if the last batch has the same length as the others.

    If not, we need to save the information separately.

    """
    if len(batches_list) == 1:
        # This is the case for a three period model.
        batches_cover_all = True
        # Set last batch info to None, because it is not needed
        last_batch_info = None
    else:
        # In the case of more periods we determine if the last two batches have equal
        # size
        batches_cover_all = len(batches_list[-1]) == len(batches_list[-2])
        # Set last batch info to None for now. If batches_cover_all is True it is not needed,
        # if it is False, it will be overwritten
        last_batch_info = None

    if not batches_cover_all:
        # In the case batches don't cover everything, we have to solve the last batch
        # separately. Delete the last element from the relevant lists and save it in
        # an extra dictionary
        last_batch = batches_list[-1]
        last_child_states_to_integrate_exog = child_states_to_integrate_stochastic_list[
            -1
        ]
        last_idx_to_aggregate_choice = child_state_choices_to_aggr_choice_list[-1]
        last_child_state_idx_interp = child_state_choice_idxs_to_interp_list[-1]

        last_state_choices = {
            key: state_choice_space[:, i][last_batch]
            for i, key in enumerate(discrete_states_names + ["choice"])
        }
        last_state_choices_childs = {
            key: state_choice_space[:, i][last_child_state_idx_interp]
            for i, key in enumerate(discrete_states_names + ["choice"])
        }
        last_parent_state_idx_of_state_choice = map_state_choice_to_parent_state[
            last_child_state_idx_interp
        ]

        last_batch_info = {
            "state_choice_idx": last_batch,
            "state_choices": last_state_choices,
            "child_states_to_integrate_stochastic": last_child_states_to_integrate_exog,
            # Child state infos.
            "child_state_choices_to_aggr_choice": last_idx_to_aggregate_choice,
            "child_state_choice_idxs_to_interp": last_child_state_idx_interp,
            "child_states_idxs": last_parent_state_idx_of_state_choice,
            "state_choices_childs": last_state_choices_childs,
        }
        batches_list = batches_list[:-1]
        child_states_to_integrate_stochastic_list = (
            child_states_to_integrate_stochastic_list[:-1]
        )
        child_state_choices_to_aggr_choice_list = (
            child_state_choices_to_aggr_choice_list[:-1]
        )
        child_state_choice_idxs_to_interp_list = child_state_choice_idxs_to_interp_list[
            :-1
        ]
    return (
        batches_list,
        child_states_to_integrate_stochastic_list,
        child_state_choices_to_aggr_choice_list,
        child_state_choice_idxs_to_interp_list,
        batches_cover_all,
        last_batch_info,
    )


def prepare_and_align_batch_arrays(
    batches_list,
    child_states_to_integrate_stochastic_list,
    child_state_choices_to_aggr_choice_list,
    child_state_choice_idxs_to_interp_list,
    state_choice_space,
    map_state_choice_to_parent_state,
    discrete_states_names,
):
    """Prepare the lists we get out of the algorithm (and after correction) for the jax
    calculations.

    They all need to have the same length of leading axis

    """
    # Get out of bound state choice idx, by taking the number of state choices + 1
    out_of_bounds_state_choice_idx = state_choice_space.shape[0] + 1

    # First convert batch information
    batch_array = np.array(batches_list)
    child_states_to_integrate_exog = np.array(child_states_to_integrate_stochastic_list)

    state_choices_batches = {
        key: state_choice_space[:, i][batch_array]
        for i, key in enumerate(discrete_states_names + ["choice"])
    }

    # Now create the child state arrays. As these can have different shapes than the
    # batches, we have to extend them:
    max_child_state_index_batch = np.max(child_states_to_integrate_exog, axis=(1, 2))
    (
        child_state_choice_idxs_to_interp,
        child_state_choices_to_aggr_choice,
    ) = extend_child_state_choices_to_aggregate_choices(
        idx_to_aggregate_choice=child_state_choices_to_aggr_choice_list,
        max_child_state_index_batch=max_child_state_index_batch,
        idx_to_interpolate=child_state_choice_idxs_to_interp_list,
        out_of_bounds_state_choice_idx=out_of_bounds_state_choice_idx,
    )
    parent_state_idx_of_state_choice = map_state_choice_to_parent_state[
        child_state_choice_idxs_to_interp
    ]
    state_choices_childs = {
        key: state_choice_space[:, i][child_state_choice_idxs_to_interp]
        for i, key in enumerate(discrete_states_names + ["choice"])
    }

    batch_info = {
        # Now the batch array information. First the batch itself
        "batches_state_choice_idx": batch_array,
        "state_choices": state_choices_batches,
        "child_states_to_integrate_stochastic": child_states_to_integrate_exog,
        # Then the child states
        "child_state_choices_to_aggr_choice": child_state_choices_to_aggr_choice,
        "child_state_choice_idxs_to_interp": child_state_choice_idxs_to_interp,
        "child_states_idxs": parent_state_idx_of_state_choice,
        "state_choices_childs": state_choices_childs,
    }
    return batch_info


def extend_child_state_choices_to_aggregate_choices(
    idx_to_aggregate_choice,
    max_child_state_index_batch,
    idx_to_interpolate,
    out_of_bounds_state_choice_idx,
):
    """In case of uneven batches, we need to extend the child state objects to cover the
    same number of state choices in each batch.

    As this object has in each batch the shape of n_state_choices x n_

    """
    # There can be also be an uneven number of child states across batches. The
    # indexes recorded in state_choice_times_exog_child_state_idxs only contain
    # the indexes up the length. So we can just fill up without of bounds indexes.
    # We also test this here
    max_n_state_unique_in_batches = list(
        map(lambda x: x.shape[0], idx_to_aggregate_choice)
    )

    # We check for internal constincy. The size (i.e. the number of states) of the
    # state_choice idx to aggregate choices in each state has to correspond to the
    # maximum state index in child indexes we integrate over.
    if not np.all(
        np.equal(
            np.array(max_n_state_unique_in_batches) - 1, max_child_state_index_batch
        )
    ):
        raise ValueError(
            "\n\nInternal error in the batch creation \n\n. "
            "Please contact developer."
        )

    # Now span an array with n_states times the maximum number of child states across
    # all batches and the number of choices. Fill with invalid state choice index
    n_batches = len(idx_to_aggregate_choice)
    max_n_child_states = np.max(max_n_state_unique_in_batches)
    n_choices = idx_to_aggregate_choice[0].shape[1]
    child_state_choices_to_aggr_choice = np.full(
        (n_batches, max_n_child_states, n_choices),
        fill_value=out_of_bounds_state_choice_idx,
        dtype=int,  # what about this hard-coded int here?
    )

    for id_batch in range(n_batches):
        child_state_choices_to_aggr_choice[
            id_batch, : max_n_state_unique_in_batches[id_batch], :
        ] = idx_to_aggregate_choice[id_batch]

    # The second array are the state choice indexes in the child states. As child
    # states can have different admissible state choices this can be different in
    # each batch. We fill up with invalid numbers.
    max_child_state_choices = np.max(list(map(len, idx_to_interpolate)))
    dummy_state = idx_to_interpolate[0][0]
    child_state_choice_idxs_to_interp = np.full(
        (n_batches, max_child_state_choices),
        fill_value=dummy_state,
        dtype=int,
    )
    for id_batch in range(n_batches):
        child_state_choice_idxs_to_interp[
            id_batch, : len(idx_to_interpolate[id_batch])
        ] = idx_to_interpolate[id_batch]

    return child_state_choice_idxs_to_interp, child_state_choices_to_aggr_choice
