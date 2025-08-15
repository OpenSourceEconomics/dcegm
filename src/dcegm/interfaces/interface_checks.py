import numpy as np
from jax import numpy as jnp


def check_states_and_choices(states, choices, model_structure):
    """Check if the states and choices are valid according to the model structure.

    Args:
        states (dict): Dictionary containing state values.
        choices (int): The choice value.
        model_structure (dict): Model structure containing information on
            discrete states and choices.

    Returns:
        bool: True if the states and choices are valid, False otherwise.

    """
    discrete_states_names = model_structure["discrete_states_names"]
    if "dummy_stochastic" in discrete_states_names:
        if "dummy_stochastic" not in states.keys():
            need_to_add_dummy = True
            # Check if all discrete states are present in states, except for the dummy stochastic state
            observed_discrete_states = list(
                set(discrete_states_names) - {"dummy_stochastic"}
            )

        else:
            need_to_add_dummy = False
            observed_discrete_states = discrete_states_names.copy()

    else:
        need_to_add_dummy = False
        observed_discrete_states = discrete_states_names.copy()

    if not all(state in states.keys() for state in observed_discrete_states):
        raise ValueError("States should contain all discrete states.")

    # We start checking the dimensions:
    # First check if the states are arrays or integers. If integers, all including choices need to be integers
    # and we convert them to arrays. Determine first dimension of choice
    if isinstance(choices, float):
        raise ValueError("Choices should be integers or arrays, not floats. ")
    # Check if choices is a single integer or numpy integers
    elif isinstance(choices, (int, np.integer)):
        choices = np.array([choices])
        single_state = True
        # Now check if all states are integers
        if not all(
            isinstance(states[key], (int, np.integer))
            for key in observed_discrete_states
        ):
            raise ValueError(
                "Discrete states should be integers or arrays. "
                "As choices is a single integer, all states must be integers as well."
            )
        else:
            states = {key: np.array([value]) for key, value in states.items()}

    elif isinstance(choices, (np.ndarray, jnp.ndarray)):
        if choices.ndim == 0:
            # Check if choices has dtype int
            if choices.dtype in [int, np.integer]:
                raise ValueError(
                    "Choices should be integers or arrays with integer dtype."
                )

            choices = np.array([choices])
            single_state = True
            # Now check if all states have dimension 0 as well
            if not all(states[key].ndim == 0 for key in states.keys()):
                raise ValueError(
                    "All states and choices must have the same dimension. Choices is dimension 0."
                )
            # All observed discrete states must have dtype int as well
            if not all(
                states[key].dtype in [int, np.integer]
                for key in observed_discrete_states
            ):
                raise ValueError(
                    "Discrete states should be integers or arrays with integer dtype. "
                )
            states = {key: np.array([value]) for key, value in states.items()}
        elif choices.ndim == 1:
            # Check if choices has dtype int
            if not np.issubdtype(choices.dtype, np.integer):
                raise ValueError(
                    "Choices should be integers or arrays with integer dtype."
                )
            single_state = False
            # Check if all states are arrays with the same dimension as choices
            if not all(
                states[key].ndim == 1 and states[key].shape[0] == choices.shape[0]
                for key in states.keys()
            ):
                raise ValueError(
                    "All states and choices must have the same dimension. Choices is dimension 1."
                )
            # All observed discrete states must have dtype int as well
            if not all(
                np.issubdtype(states[key].dtype, np.integer)
                for key in observed_discrete_states
            ):
                raise ValueError(
                    "Discrete states should be integers or arrays with integer dtype. "
                )
        else:
            raise ValueError(
                "Choices should be integers or arrays with dimension 0 or 1."
            )
    else:
        raise ValueError("Choices should be integers or arrays with dimension 0 or 1.")

    if need_to_add_dummy:
        if single_state:
            states["dummy_stochastic"] = np.array([0])
        else:
            states["dummy_stochastic"] = np.zeros(choices.shape[0], dtype=int)

    state_choices = {
        **states,
        "choice": choices,
    }
    return state_choices
