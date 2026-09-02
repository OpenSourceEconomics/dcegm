import numpy as np


def add_last_two_period_information(
    n_periods,
    model_structure,
):
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

    # Select state_choice idxs in final period
    idx_state_choice_final_period = np.where(state_choice_space[:, 0] == n_periods - 1)[
        0
    ]
    # To solve the second last period, we need the child states in the last period
    # and the corresponding matrix, where each row is a state with the state choice
    # ids as entry in each choice
    idx_states_final_period = np.where(state_space[:, 0] == n_periods - 1)[0]
    states_final_period = state_space[idx_states_final_period]
    # Now construct a tuple for indexing
    n_state_vars = states_final_period.shape[1]
    states_tuple = tuple(states_final_period[:, i] for i in range(n_state_vars))

    # Now get the matrix we use for choice aggregation
    state_to_choices_final_period = map_state_choice_to_index[states_tuple]

    # Reindex the array, which maps state choices to states in the final period.
    # We need to do that, so we can index in idx_state_choice_final_period.
    # As the state choice space is ordered by period,
    # idx_state_choice_final_period are consecutive integers, with the first entry
    # being the smallest. By subtracting the minimum we get the reindexing.
    min_val = int(np.min(idx_state_choice_final_period))
    state_to_choices_final_period -= min_val

    idx_state_choice_second_last_period = np.where(
        state_choice_space[:, 0] == n_periods - 2
    )[0]

    # Now turn to the child state indexes
    child_states_second_last_period = map_state_choice_to_child_states[
        idx_state_choice_second_last_period
    ]

    # Reindex the child states of the second last period, i.e. child_states_second_last_period
    # from indexes of the whole state space to indexes of idx_states_final_period.
    # As the state space is ordered by period, idx_states_final_period
    # are consecutive integers. It could be that not all states in the final period are child states
    # of the second last period (imagine a death state, all accumulated in the final period).
    # Therefore, we reindex with the smallest index in idx_states_final_period. So if the child states
    # do cover all states child_states_second_last_period, will have a 0 as minimal number.
    min_val = int(np.min(idx_states_final_period))
    child_states_second_last_period -= min_val

    # Also add parent states in last period
    parent_states_final_period = map_state_choice_to_parent_state[
        idx_state_choice_final_period
    ]

    # A representative second-to-last-period *state-choice* for each final-period
    # state-choice -- used only to pick which state-choice's own continuous grid
    # feeds the law of motion in solve_final_period (see law_of_motion.py). Grids
    # live on the state-choice space (that's where the solution itself lives), so
    # the representative parent is a state-choice index, not a bare state. Mirrors
    # the same computation the main backward induction loop does per batch (see
    # child_state_dedup.py), specialized for the (undeduplicated) two-period case:
    # solve_final_period computes the law of motion for every final-period
    # state-choice individually, so we need a representative parent aligned with
    # every entry of idx_state_choice_final_period, not just a deduplicated subset.
    child_states_idxs_from_second_last = map_state_choice_to_child_states[
        idx_state_choice_second_last_period
    ]
    n_stochastic_states = child_states_idxs_from_second_last.shape[1]
    unique_final_states, first_occurrence_flat = np.unique(
        child_states_idxs_from_second_last, return_index=True
    )
    representative_second_last_row = first_occurrence_flat // n_stochastic_states
    # idx_state_choice_second_last_period[...] is already a state-choice index --
    # exactly the true (previous-period) parent state-choice we want, no need to
    # collapse it down to a bare state.
    representative_parent_state_choice_of_unique_final_state = (
        idx_state_choice_second_last_period[representative_second_last_row]
    )

    invalid_state_idx = np.iinfo(map_state_choice_to_child_states.dtype).max
    valid_mask = unique_final_states != invalid_state_idx

    # parent_states_final_period (each final-period state-choice's own bare state)
    # is used purely as a lookup key here, matching each final-period row to its
    # corresponding entry in unique_final_states -- unrelated to grid selection.
    # unique_final_states is sorted (np.unique guarantees this), so this is a
    # vectorized lookup via searchsorted; any final-period state not actually
    # reachable from the second-to-last period (e.g. an absorbing/initial-like
    # state; already flagged elsewhere via a warning in test_child_state_mapping,
    # not an error) falls back to its own state-choice identity, mirroring
    # calc_law_of_motion_for_state_choices's own
    # grid_source_state_choice_vec=None -> self fallback.
    insert_pos = np.clip(
        np.searchsorted(unique_final_states, parent_states_final_period),
        0,
        len(unique_final_states) - 1,
    )
    found_mask = (
        unique_final_states[insert_pos] == parent_states_final_period
    ) & valid_mask[insert_pos]
    representative_second_last_period_parent_idx_for_final_period = np.where(
        found_mask,
        representative_parent_state_choice_of_unique_final_state[insert_pos],
        idx_state_choice_final_period,
    )

    last_two_period_info = {
        "idx_state_choices_final_period": idx_state_choice_final_period,
        "idx_state_choices_second_last_period": idx_state_choice_second_last_period,
        "idxs_parent_states_final_period": parent_states_final_period,
        "state_to_choices_final_period": state_to_choices_final_period,
        "child_states_second_last_period": child_states_second_last_period,
        "representative_second_last_period_parent_idx_for_final_period": (
            representative_second_last_period_parent_idx_for_final_period
        ),
    }

    state_choice_space_dict = model_structure["state_choice_space_dict"]
    # Also add state choice mat as dictionary for each of the two periods
    for idx, period_name in [
        (idx_state_choice_final_period, "final"),
        (idx_state_choice_second_last_period, "second_last"),
    ]:
        last_two_period_info[f"state_choice_mat_{period_name}_period"] = {
            key: var[idx] for key, var in state_choice_space_dict.items()
        }
    return last_two_period_info
