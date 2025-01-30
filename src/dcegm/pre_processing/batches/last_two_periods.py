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

    last_two_period_info = {
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
        last_two_period_info[f"state_choice_mat_{period_name}_period"] = {
            key: state_choice_space[:, i][idx]
            for i, key in enumerate(discrete_states_names + ["choice"])
        }
    return last_two_period_info
