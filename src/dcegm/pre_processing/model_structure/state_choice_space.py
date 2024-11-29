import numpy as np

from dcegm.pre_processing.model_structure.checks import check_endog_update_function
from dcegm.pre_processing.model_structure.shared import create_indexer_for_space
from dcegm.pre_processing.shared import get_smallest_int_type


def create_state_choice_space_and_child_state_mapping(
    state_space_options,
    get_state_specific_choice_set,
    get_next_period_state,
    dict_of_state_space_objects,
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

    states_names_without_exog = dict_of_state_space_objects["state_names_without_exog"]
    exog_state_names = dict_of_state_space_objects["exog_states_names"]
    exog_state_space = dict_of_state_space_objects["exog_state_space"]
    map_child_state_to_index = dict_of_state_space_objects["map_child_state_to_index"]
    map_state_to_index = dict_of_state_space_objects["map_state_to_index"]
    state_space = dict_of_state_space_objects["state_space"]

    n_states, n_state_and_exog_variables = state_space.shape
    n_exog_states, n_exog_vars = exog_state_space.shape
    n_choices = len(state_space_options["choices"])
    discrete_states_names = states_names_without_exog + exog_state_names
    n_periods = state_space_options["n_periods"]

    dtype_exog_state_space = get_smallest_int_type(n_exog_states)

    # Get dtype and maxint for choices
    dtype_choices = get_smallest_int_type(n_choices)
    # Get dtype and max int for state space
    state_space_dtype = state_space.dtype

    if np.iinfo(state_space_dtype).max > np.iinfo(dtype_choices).max:
        state_choice_space_dtype = state_space_dtype
    else:
        state_choice_space_dtype = dtype_choices

    state_choice_space_raw = np.zeros(
        (n_states * n_choices, n_state_and_exog_variables + 1),
        dtype=state_choice_space_dtype,
    )

    state_space_indexer_dtype = map_child_state_to_index.dtype
    invalid_indexer_idx = np.iinfo(state_space_indexer_dtype).max

    map_state_choice_to_parent_state = np.zeros(
        (n_states * n_choices), dtype=state_space_indexer_dtype
    )

    map_state_choice_to_child_states = np.full(
        (n_states * n_choices, n_exog_states),
        fill_value=invalid_indexer_idx,
        dtype=state_space_indexer_dtype,
    )

    exog_states_tuple = tuple(exog_state_space[:, i] for i in range(n_exog_vars))

    idx = 0
    for state_vec in state_space:
        state_idx = map_state_to_index[tuple(state_vec)]

        # Full state dictionary
        this_period_state = {
            key: state_vec[i] for i, key in enumerate(discrete_states_names)
        }

        feasible_choice_set = get_state_specific_choice_set(
            **this_period_state,
        )

        for choice in feasible_choice_set:
            state_choice_space_raw[idx, :-1] = state_vec
            state_choice_space_raw[idx, -1] = choice

            map_state_choice_to_parent_state[idx] = state_idx

            if state_vec[0] < n_periods - 1:

                endog_state_update = get_next_period_state(
                    **this_period_state, choice=choice
                )

                check_endog_update_function(
                    endog_state_update, this_period_state, choice, exog_state_names
                )

                next_period_state = this_period_state.copy()
                next_period_state.update(endog_state_update)

                next_period_state_tuple_wo_exog = tuple(
                    np.full(
                        n_exog_states,
                        fill_value=next_period_state[key],
                        dtype=dtype_exog_state_space,
                    )
                    for key in states_names_without_exog
                )

                states_next_tuple = next_period_state_tuple_wo_exog + exog_states_tuple

                try:
                    child_idxs = map_child_state_to_index[states_next_tuple]
                except:
                    raise IndexError(
                        f"\n\n The state \n\n{endog_state_update}\n\n is reached as a "
                        f"child state from an existing state, but does not exist for "
                        f"some values of the exogenous processes. Please check if it "
                        f"should not be reached or should exist by adapting the "
                        f"sparsity condition and/or the set of possible state values."
                    )

                map_state_choice_to_child_states[idx, :] = child_idxs

            idx += 1

    state_choice_space = state_choice_space_raw[:idx]
    map_state_choice_to_index = create_indexer_for_space(state_choice_space)

    state_choice_space_dict = {
        key: state_choice_space[:, i]
        for i, key in enumerate(discrete_states_names + ["choice"])
    }

    dict_of_state_choice_space_objects = {
        "state_choice_space": state_choice_space,
        "state_choice_space_dict": state_choice_space_dict,
        "map_state_choice_to_index": map_state_choice_to_index,
        "map_state_choice_to_parent_state": map_state_choice_to_parent_state[:idx],
        "map_state_choice_to_child_states": map_state_choice_to_child_states[:idx, :],
    }

    return dict_of_state_choice_space_objects
