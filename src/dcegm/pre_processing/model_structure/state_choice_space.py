import warnings

import numpy as np

from dcegm.pre_processing.model_structure.shared import create_indexer_for_space
from dcegm.pre_processing.shared import get_smallest_int_type


def create_state_choice_space_and_child_state_mapping(
    model_config,
    state_specific_choice_set,
    next_period_deterministic_state,
    state_space_arrays,
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
        state_specific_choice_set (Callable): User-supplied function that returns
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

    state_names_without_stochastic = state_space_arrays[
        "state_names_without_stochastic"
    ]
    stochastic_states_names = state_space_arrays["stochastic_states_names"]
    stochastic_state_space = state_space_arrays["stochastic_state_space"]
    map_state_to_index_with_proxy = state_space_arrays["map_state_to_index_with_proxy"]
    map_state_to_index = state_space_arrays["map_state_to_index"]
    state_space = state_space_arrays["state_space"]

    n_states, n_state_and_stochastic_variables = state_space.shape
    n_stochastic_states, n_stochastic_vars = stochastic_state_space.shape
    n_choices = len(model_config["choices"])
    discrete_states_names = state_names_without_stochastic + stochastic_states_names
    n_periods = model_config["n_periods"]

    dtype_stochastic_state_space = get_smallest_int_type(n_stochastic_states)

    # Get dtype and maxint for choices
    dtype_choices = get_smallest_int_type(n_choices)
    # Get dtype and max int for state space
    state_space_dtype = state_space.dtype

    if np.iinfo(state_space_dtype).max > np.iinfo(dtype_choices).max:
        state_choice_space_dtype = state_space_dtype
    else:
        state_choice_space_dtype = dtype_choices

    state_choice_space_raw = np.zeros(
        (n_states * n_choices, n_state_and_stochastic_variables + 1),
        dtype=state_choice_space_dtype,
    )

    state_space_indexer_dtype = map_state_to_index_with_proxy.dtype
    invalid_indexer_idx = np.iinfo(state_space_indexer_dtype).max

    map_state_choice_to_parent_state = np.zeros(
        (n_states * n_choices), dtype=state_space_indexer_dtype
    )

    map_state_choice_to_child_states = np.full(
        (n_states * n_choices, n_stochastic_states),
        fill_value=invalid_indexer_idx,
        dtype=state_space_indexer_dtype,
    )

    stochastic_states_tuple = tuple(
        stochastic_state_space[:, i] for i in range(n_stochastic_vars)
    )

    idx = 0
    for state_vec in state_space:
        state_idx = map_state_to_index[tuple(state_vec)]

        # Full state dictionary
        this_period_state = {
            key: state_vec[i] for i, key in enumerate(discrete_states_names)
        }

        feasible_choice_set = state_specific_choice_set(
            **this_period_state,
        )

        for choice in feasible_choice_set:
            state_choice_space_raw[idx, :-1] = state_vec
            state_choice_space_raw[idx, -1] = choice

            map_state_choice_to_parent_state[idx] = state_idx

            if state_vec[0] < n_periods - 1:

                endog_state_update = next_period_deterministic_state(
                    **this_period_state, choice=choice
                )

                check_endog_update_function(
                    endog_state_update,
                    this_period_state,
                    choice,
                    stochastic_states_names,
                )

                next_period_state = this_period_state.copy()
                next_period_state.update(endog_state_update)

                next_period_state_tuple_wo_stochastic = tuple(
                    np.full(
                        n_stochastic_states,
                        fill_value=next_period_state[key],
                        dtype=dtype_stochastic_state_space,
                    )
                    for key in state_names_without_stochastic
                )

                states_next_tuple = (
                    next_period_state_tuple_wo_stochastic + stochastic_states_tuple
                )

                try:
                    child_idxs = map_state_to_index_with_proxy[states_next_tuple]
                except:
                    raise IndexError(
                        f"\n\n The state \n\n{endog_state_update}\n\n is a child state of "
                        f"the state-choice combination \n\n{this_period_state}\n\n with choice: "
                        f"{choice}.\n\n The state variables are out of bounds for the defined state space "
                        f"Please check the possible state values in the state space definition."
                    )

                invalid_child_state_idxs = np.where(child_idxs == invalid_indexer_idx)[
                    0
                ]
                if len(invalid_child_state_idxs) > 0:
                    invalid_child_state_example = np.array(states_next_tuple).T[
                        invalid_child_state_idxs[0]
                    ]
                    invalid_child_state_dict = {
                        key: invalid_child_state_example[i]
                        for i, key in enumerate(discrete_states_names)
                    }
                    warnings.warn(
                        f"\n\n The state \n\n{invalid_child_state_dict}\n\n is a child state of "
                        f"the state \n\n{this_period_state}\n\n with choice: {choice}.\n\n "
                        f"It is also declared invalid by the sparsity condition. Please "
                        f"remember, that if a state is invalid because it can't be reached by the deterministic"
                        f"update of states, this has to be reflected in the state space function next_period_deterministic_state."
                        f"If its stochastic state realization is invalid, this state has to be proxied to another state"
                        f"by the sparsity condition."
                    )

                map_state_choice_to_child_states[idx, :] = child_idxs

            idx += 1

    # Select only needed rows of arrays
    state_choice_space = state_choice_space_raw[:idx]
    map_state_choice_to_parent_state = map_state_choice_to_parent_state[:idx]
    map_state_choice_to_child_states = map_state_choice_to_child_states[:idx, :]

    map_state_choice_to_index, _ = create_indexer_for_space(state_choice_space)

    # Create indexer with proxy
    state_space_incl_proxies = state_space_arrays["state_space_incl_proxies"]
    state_space_incl_proxies_tuple = tuple(
        state_space_incl_proxies[:, i] for i in range(n_state_and_stochastic_variables)
    )
    states_proxy_to = state_space[
        map_state_to_index_with_proxy[state_space_incl_proxies_tuple]
    ]
    states_proxy_to_tuple = tuple(
        states_proxy_to[:, i] for i in range(n_state_and_stochastic_variables)
    )
    map_state_choice_to_index_with_proxy = np.empty_like(map_state_choice_to_index)
    map_state_choice_to_index_with_proxy[state_space_incl_proxies_tuple] = (
        map_state_choice_to_index[states_proxy_to_tuple]
    )

    state_choice_space_dict = {
        key: state_choice_space[:, i]
        for i, key in enumerate(discrete_states_names + ["choice"])
    }

    test_child_state_mapping(
        model_config=model_config,
        state_choice_space=state_choice_space,
        state_space=state_space,
        map_state_choice_to_child_states=map_state_choice_to_child_states,
        discrete_states_names=discrete_states_names,
    )

    dict_of_state_choice_space_objects = {
        "state_choice_space": state_choice_space,
        "state_choice_space_dict": state_choice_space_dict,
        "map_state_choice_to_index": map_state_choice_to_index,
        "map_state_choice_to_index_with_proxy": map_state_choice_to_index_with_proxy,
        "map_state_choice_to_parent_state": map_state_choice_to_parent_state,
        "map_state_choice_to_child_states": map_state_choice_to_child_states,
    }

    return dict_of_state_choice_space_objects


def test_child_state_mapping(
    model_config,
    state_choice_space,
    state_space,
    map_state_choice_to_child_states,
    discrete_states_names,
):
    """Test state space objects for consistency."""
    n_periods = model_config["n_periods"]
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
        warnings.warn(
            f"\n\n\n\n Some states are not child states of any state-choice "
            f"combination or stochastic transition. Please revisit the sparsity condition. \n \n"
            f"An example of a state that is not a child state is: \n \n"
            f"{example_dict} \n \n"
        )


def check_endog_update_function(
    endog_state_update, this_period_state, choice, stochastic_state_names
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

    # Check if stochastic state is updated. This is forbidden.
    for state_name in stochastic_state_names:
        if state_name in endog_state_update.keys():
            raise ValueError(
                f"\n\n The stochastic state {state_name} is also updated (or just returned)"
                f"for in the endogenous update function. You can use the proxy function to implement"
                f"a custom update rule, i.e. redirecting the stochastic state."
                f"An example of this update happens with the state choice combination: \n\n"
                f"{this_period_state} \n\n"
            )
