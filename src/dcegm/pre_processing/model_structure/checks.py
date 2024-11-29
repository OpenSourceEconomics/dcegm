import numpy as np


def check_endog_update_function(
    endog_state_update, this_period_state, choice, exog_state_names
):
    """Conduct several checks on the endogenous state update function."""
    if endog_state_update["period"] != this_period_state["period"] + 1:
        raise ValueError(
            f"\n\n The update function does not return the correct next period count."
            f"An example of this update happens with the state choice combination: \n\n"
            f"{this_period_state} \n\n"
        )

    if endog_state_update["lagged_choice"] != choice:
        raise ValueError(
            f"\n\n The update function does not return the correct lagged choice for a given choice."
            f"An example of this update happens with the state choice combination: \n\n"
            f"{this_period_state} \n\n"
        )

    # Check if exogenous state is updated. This is forbidden.
    for exog_state_name in exog_state_names:
        if exog_state_name in endog_state_update.keys():
            raise ValueError(
                f"\n\n The exogenous state {exog_state_name} is also updated (or just returned)"
                f"for in the endogenous update function. You can use the proxy function to implement"
                f"a custom update rule, i.e. redirecting the exogenous process."
                f"An example of this update happens with the state choice combination: \n\n"
                f"{this_period_state} \n\n"
            )


def test_child_state_mapping(
    state_space_options,
    state_choice_space,
    state_space,
    map_state_choice_to_child_states,
    discrete_states_names,
):
    """Test state space objects for consistency."""
    n_periods = state_space_options["n_periods"]
    state_choices_idxs_wo_last = np.where(state_choice_space[:, 0] < n_periods - 1)[0]

    # Check if all feasible state choice combinations have a valid child state
    idxs_child_states = map_state_choice_to_child_states[state_choices_idxs_wo_last, :]

    # Get dtype and max int for state space indexer
    state_space_indexer_dtype = map_state_choice_to_child_states.dtype
    invalid_state_space_idx = np.iinfo(state_space_indexer_dtype).max

    if np.any(idxs_child_states == invalid_state_space_idx):
        # Get row axis of child states that are invalid
        invalid_child_states = np.unique(
            np.where(idxs_child_states == invalid_state_space_idx)[0]
        )
        invalid_state_choices_example = state_choice_space[invalid_child_states[0]]
        example_dict = {
            key: invalid_state_choices_example[i]
            for i, key in enumerate(discrete_states_names)
        }
        example_dict["choice"] = invalid_state_choices_example[-1]
        raise ValueError(
            f"\n\n\n\n Some state-choice combinations have invalid child "
            f"states. Please update accordingly the deterministic law of motion or"
            f"the proxy function."
            f"\n \n An example of a combination of state and choice with "
            f"invalid child states is: \n \n"
            f"{example_dict} \n \n"
        )

    # Check if all states are a child states except the ones in the first period
    idxs_states_except_first = np.where(state_space[:, 0] > 0)[0]
    idxs_states_except_first_in_child_states = np.isin(
        idxs_states_except_first, idxs_child_states
    )
    if not np.all(idxs_states_except_first_in_child_states):
        not_child_state_idxs = idxs_states_except_first[
            ~idxs_states_except_first_in_child_states
        ]
        not_child_state_example = state_space[not_child_state_idxs[0]]
        example_dict = {
            key: not_child_state_example[i]
            for i, key in enumerate(discrete_states_names)
        }
        raise ValueError(
            f"\n\n\n\n Some states are not child states of any state-choice "
            f"combination or stochastic transition. Please revisit the sparsity condition. \n \n"
            f"An example of a state that is not a child state is: \n \n"
            f"{example_dict} \n \n"
        )
