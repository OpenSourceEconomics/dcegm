def get_child_state_index_per_state_choice(states, choice, model_structure):
    states_choice_dict = {**states, "choice": choice}
    state_choice_index = get_state_choice_index_per_discrete_state_and_choice(
        model_structure, states_choice_dict
    )

    child_states = model_structure["map_state_choice_to_child_states"][
        state_choice_index
    ]

    return child_states


def get_state_choice_index_per_discrete_state(
    states, map_state_choice_to_index, discrete_states_names
):
    """Get the state-choice index for a given set of discrete states.

    Args:
        map_state_choice_to_index (dict): Mapping from a state-choice tuple to
            an index.
        states (dict): Dictionary of state values.
        discrete_states_names (list[str]): Names of discrete state variables.

    Returns:
        int: The index corresponding to the given discrete states.

    """
    indexes = map_state_choice_to_index[
        tuple((states[key],) for key in discrete_states_names)
    ]
    # As the code above generates a dummy dimension in the first index, remove it
    return indexes[0]


def get_state_choice_index_per_discrete_state_and_choice(
    model_structure, state_choice_dict
):
    """Get the state-choice index for a given set of discrete states and a choice.

    Args:
        model (dict): A dictionary representing the model. Must contain
            'model_structure' with a 'map_state_choice_to_index_with_proxy'
            and 'discrete_states_names'.
        state_choice_dict (dict): Dictionary containing discrete states and
            the choice.

    Returns:
        int: The index corresponding to the specified discrete states and choice.

    """
    map_state_choice_to_index = model_structure["map_state_choice_to_index_with_proxy"]
    discrete_states_names = model_structure["discrete_states_names"]
    state_choice_tuple = tuple(
        state_choice_dict[st] for st in discrete_states_names + ["choice"]
    )
    state_choice_index = map_state_choice_to_index[state_choice_tuple]

    return state_choice_index
