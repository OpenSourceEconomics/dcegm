import numpy as np


def get_child_state_index_per_states_and_choices(states, choices, model_structure):
    state_choice_index = get_state_choice_index_per_discrete_states_and_choices(
        model_structure, states, choices
    )

    child_states = model_structure["map_state_choice_to_child_states"][
        state_choice_index
    ]

    return child_states


def get_state_choice_index_per_discrete_states(
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
    # Need flag to only evaluate in non jit mode
    # max_values_per_state = {key: np.max(states[key]) for key in discrete_states_names}
    # # Check that max value does not exceed the dimension
    # dim = map_state_choice_to_index.shape
    # for i, key in enumerate(discrete_states_names):
    #     if max_values_per_state[key] > dim[i] - 1:
    #         raise ValueError(
    #             f"Max value of state {key} exceeds the dimension of the model."
    #         )

    # As the code above generates a dummy dimension in the first index, remove it
    return indexes[0]


def get_state_choice_index_per_discrete_states_and_choices(
    model_structure, states, choices
):
    """Get the state-choice index for a given set of discrete states and a choice.

    Args:
        model_structure (dict): Model structure containing all information on the structure of the model.
        states (dict): Dictionary containing discrete states and
            the choice.

    Returns:
        int: The index corresponding to the specified discrete states and choice.

    """
    state_choices = {"choice": choices, **states}

    map_state_choice_to_index = model_structure["map_state_choice_to_index_with_proxy"]
    discrete_states_names = model_structure["discrete_states_names"]
    state_choice_tuple = tuple(
        state_choices[st] for st in discrete_states_names + ["choice"]
    )
    state_choice_index = map_state_choice_to_index[state_choice_tuple]

    return state_choice_index
