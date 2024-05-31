"""Functions for creating internal state space objects."""
import jax.numpy as jnp
import numpy as np
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options


def create_state_space_and_choice_objects(
    options,
    model_funcs,
):
    """Create dictionary of state and state-choice objects for each period.

    Args:
        options (Dict[str, int]): Options dictionary.

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
        state_space_dict,
        map_state_to_index,
        states_names_without_exog,
        exog_states_names,
        exog_state_space,
    ) = create_state_space(options)

    (
        state_choice_space,
        map_state_choice_to_index,
        map_state_choice_to_parent_state,
        map_state_choice_to_child_states,
    ) = create_state_choice_space(
        state_space_options=options["state_space"],
        state_space=state_space,
        states_names_without_exog=states_names_without_exog,
        exog_state_names=exog_states_names,
        exog_state_space=exog_state_space,
        map_state_to_index=map_state_to_index,
        get_state_specific_choice_set=model_funcs["get_state_specific_choice_set"],
        get_next_period_state=model_funcs["get_next_period_state"],
    )

    test_state_space_objects(
        state_space_options=options["state_space"],
        state_choice_space=state_choice_space,
        map_state_choice_to_child_states=map_state_choice_to_child_states,
        state_space_names=states_names_without_exog + exog_states_names,
    )

    model_structure = {
        "state_space": state_space,
        "choice_range": jnp.asarray(options["state_space"]["choices"]),
        "state_space_dict": state_space_dict,
        "map_state_to_index": map_state_to_index,
        "exog_state_space": exog_state_space,
        "states_names_without_exog": states_names_without_exog,
        "exog_states_names": exog_states_names,
        "state_space_names": states_names_without_exog + exog_states_names,
        "state_choice_space": state_choice_space,
        "map_state_choice_to_index": map_state_choice_to_index,
        "map_state_choice_to_parent_state": map_state_choice_to_parent_state,
        "map_state_choice_to_child_states": map_state_choice_to_child_states,
    }

    return model_structure


def test_state_space_objects(
    state_space_options,
    state_choice_space,
    map_state_choice_to_child_states,
    state_space_names,
):
    """Test state space objects for consistency."""
    n_periods = state_space_options["n_periods"]
    state_choices_idxs_wo_last = np.where(state_choice_space[:, 0] < n_periods - 1)[0]

    # Check if all feasible state choice combinations have a valid child state
    child_states = map_state_choice_to_child_states[state_choices_idxs_wo_last, :]
    if not np.all(child_states >= 0):
        # Get row axis of child states that are invalid
        invalid_child_states = np.unique(np.where(child_states < 0)[0])
        invalid_state_choices_example = state_choice_space[invalid_child_states[0]]
        example_dict = {
            key: invalid_state_choices_example[i]
            for i, key in enumerate(state_space_names)
        }
        example_dict["choice"] = invalid_state_choices_example[-1]
        raise ValueError(
            f"\n\n\n\n Some state-choice combinations have invalid child "
            f"states. "
            f"\n \n An example of a combination of state and choice with "
            f"invalid child states is: \n \n"
            f"{example_dict} \n \n"
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
        _num_states_of_all_endog_states,
        num_endog_states,
        sparsity_func,
    ) = process_endog_state_specifications(
        state_space_options=state_space_options, model_params=model_params
    )

    (
        exog_states_names,
        _num_states_of_all_exog_states,
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

    n_exog_states = exog_state_space.shape[0]

    state_space_wo_exog = np.array(state_space_wo_exog_list)
    state_space_wo_exog_full = np.repeat(state_space_wo_exog, n_exog_states, axis=0)
    exog_state_space_full = np.tile(exog_state_space, (state_space_wo_exog.shape[0], 1))
    state_space = np.concatenate(
        (state_space_wo_exog_full, exog_state_space_full), axis=1
    )

    dtype_state_space = get_smallest_uint_type(state_space.shape[0])
    max_int_state_space = np.iinfo(dtype_state_space).max

    # Create indexer array that maps states to indexes
    map_state_to_index = create_indexer_for_space(
        state_space, dtype_state_space, max_int_state_space
    )

    state_space_dict = {
        key: state_space[:, i]
        for i, key in enumerate(states_names_without_exog + exog_states_names)
    }

    return (
        state_space,
        state_space_dict,
        map_state_to_index,
        states_names_without_exog,
        exog_states_names,
        exog_state_space,
    )


def create_state_choice_space(
    state_space_options,
    state_space,
    exog_state_space,
    states_names_without_exog,
    exog_state_names,
    map_state_to_index,
    get_state_specific_choice_set,
    get_next_period_state,
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
    n_exog_states, n_exog_vars = exog_state_space.shape
    n_choices = len(state_space_options["choices"])
    state_space_names = states_names_without_exog + exog_state_names
    n_periods = state_space_options["n_periods"]

    dtype_exog_state_space = get_smallest_uint_type(n_exog_states)
    dtype_state_choice_space = get_smallest_uint_type(n_states * n_choices)
    max_int_state_choice_space = np.iinfo(dtype_state_choice_space).max

    state_choice_space = np.zeros(
        (n_states * n_choices, n_state_and_exog_variables + 1),
        dtype=dtype_state_choice_space,
    )
    map_state_choice_to_parent_state = np.zeros(
        (n_states * n_choices), dtype=dtype_state_choice_space
    )
    map_state_choice_to_child_states = np.full(
        (n_states * n_choices, n_exog_states),
        fill_value=max_int_state_choice_space,
        dtype=dtype_state_choice_space,
    )

    exog_states_tuple = tuple(exog_state_space[:, i] for i in range(n_exog_vars))

    idx = 0
    for state_vec in state_space:
        state_idx = map_state_to_index[tuple(state_vec)]

        # Full state dictionary
        state_dict = {key: state_vec[i] for i, key in enumerate(state_space_names)}

        feasible_choice_set = get_state_specific_choice_set(
            **state_dict,
        )

        for choice in feasible_choice_set:
            state_choice_space[idx, :-1] = state_vec
            state_choice_space[idx, -1] = choice

            map_state_choice_to_parent_state[idx] = state_idx

            if state_vec[0] < n_periods - 1:
                # Current state without exog
                state_dict_without_exog = {
                    key: state_dict[key]
                    for i, key in enumerate(states_names_without_exog)
                }

                endog_state_update = get_next_period_state(
                    **state_dict_without_exog, choice=choice
                )

                state_dict_without_exog.update(endog_state_update)

                states_next_tuple = (
                    tuple(
                        np.full(
                            n_exog_states,
                            fill_value=state_dict_without_exog[key],
                            dtype=dtype_exog_state_space,
                        )
                        for key in states_names_without_exog
                    )
                    + exog_states_tuple
                )
                try:
                    child_idxs = map_state_to_index[states_next_tuple]
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

    state_choice_space_final = state_choice_space[:idx]
    map_state_choice_to_index = create_indexer_for_space(
        state_choice_space_final, dtype_state_choice_space, max_int_state_choice_space
    )

    return (
        state_choice_space_final,
        map_state_choice_to_index,
        map_state_choice_to_parent_state[:idx],
        map_state_choice_to_child_states[:idx, :],
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

    else:
        exog_state_names = ["dummy_exog"]
        num_states_of_all_exog_states = [1]

        exog_state_space = np.array([[0]], dtype=np.uint8)

    return (
        exog_state_names,
        num_states_of_all_exog_states,
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


def create_indexer_for_space(space, dtype_state_space, max_int_state_space):
    """Creates indexer for spaces.

    We need to think about which datatype we want to use and what is our invalid number.
    Who doesn't like -99999999? Will anybody ever have 10 Billion state choices.

    """

    data_type = get_smallest_uint_type(space.shape[0])
    max_value = np.iinfo(data_type).max

    max_var_values = np.max(space, axis=0)
    map_vars_to_index = np.full(
        max_var_values + 1, fill_value=max_value, dtype=data_type
    )
    index_tuple = tuple(space[:, i] for i in range(space.shape[1]))

    map_vars_to_index[index_tuple] = np.arange(space.shape[0], dtype=data_type)

    return map_vars_to_index


def check_options(options):
    """Check if options are valid."""
    if not isinstance(options, dict):
        raise ValueError("Options must be a dictionary.")

    if "state_space" not in options:
        raise ValueError("Options must contain a state space dictionary.")

    if not isinstance(options["state_space"], dict):
        raise ValueError("State space must be a dictionary.")

    if "n_periods" not in options["state_space"]:
        raise ValueError("State space must contain the number of periods.")

    if not isinstance(options["state_space"]["n_periods"], int):
        raise ValueError("Number of periods must be an integer.")

    if "choices" not in options["state_space"]:
        print("Choices not given. Assume only single choice with value 0")
        options["state_space"]["choices"] = np.array([0], dtype=np.uint8)

    if "model_params" not in options:
        raise ValueError("Options must contain a model parameters dictionary.")

    if not isinstance(options["model_params"], dict):
        raise ValueError("Model parameters must be a dictionary.")

    return options


def get_smallest_uint_type(n_values):
    uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in uint_types:
        if np.iinfo(dtype).max >= n_values:
            return dtype
