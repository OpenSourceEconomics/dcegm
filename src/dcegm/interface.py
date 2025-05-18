"""Interface functions."""

import jax
import jax.numpy as jnp
import pandas as pd

from dcegm.interpolation.interp1d import (
    interp1d_policy_and_value_on_wealth,
    interp_policy_on_wealth,
    interp_value_on_wealth,
)
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
    interp2d_policy_on_wealth_and_regular_grid,
    interp2d_value_on_wealth_and_regular_grid,
)


def get_n_state_choice_period(model):
    """Get the number of state-choice periods from the model.

    Args:
        model (dict): A dictionary containing model information. Must include
            'model_structure' with a 'state_choice_space' key.

    Returns:
        pd.Series: A pandas Series with value counts of the first column of
        'state_choice_space', sorted by index.
    """
    return (
        pd.Series(model["model_structure"]["state_choice_space"][:, 0])
        .value_counts()
        .sort_index()
    )


def policy_and_value_for_state_choice_vec(
    endog_grid_solved,
    value_solved,
    policy_solved,
    params,
    model,
    state_choice_vec,
    wealth,
    compute_utility,
    second_continuous=None,
):
    """Get policy and value for a given state and choice vector.

    Args:
        endog_grid_solved (jnp.ndarray): Endogenous wealth grid for all states
            and choices.
        value_solved (jnp.ndarray): Value array for all states and choices.
        policy_solved (jnp.ndarray): Policy array for all states and choices.
        params (dict): Dictionary containing model parameters.
        model (dict): Dictionary containing model information and settings.
        state_choice_vec (dict): Dictionary containing a single state and choice.
        wealth (float): The wealth level at which to interpolate.
        compute_utility (Callable): Function to compute utility given the state,
            choice, and parameters.
        second_continuous (float, optional): An additional continuous state
            dimension. If provided, interpolation is done in two dimensions.

    Returns:
        Tuple[float, float]: A tuple of (policy, value) at the given state and
        choice.

    """
    map_state_choice_to_index = model["model_structure"][
        "map_state_choice_to_index_with_proxy"
    ]
    discrete_states_names = model["model_structure"]["discrete_states_names"]

    state_choice_tuple = tuple(
        state_choice_vec[st] for st in discrete_states_names + ["choice"]
    )
    state_choice_index = map_state_choice_to_index[state_choice_tuple]
    continuous_states_info = model["model_config"]["continuous_states_info"]

    if second_continuous is None:
        policy, value = interp1d_policy_and_value_on_wealth(
            wealth=wealth,
            endog_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
            policy=jnp.take(policy_solved, state_choice_index, axis=0),
            value=jnp.take(value_solved, state_choice_index, axis=0),
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
        )
    else:
        policy, value = interp2d_policy_and_value_on_wealth_and_regular_grid(
            regular_grid=continuous_states_info["second_continuous_grid"],
            wealth_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
            value_grid=jnp.take(value_solved, state_choice_index, axis=0),
            policy_grid=jnp.take(policy_solved, state_choice_index, axis=0),
            regular_point_to_interp=second_continuous,
            wealth_point_to_interp=wealth,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
        )

    return policy, value


def value_for_state_choice_vec(
    endog_grid_solved,
    value_solved,
    params,
    model,
    state_choice_vec,
    wealth,
    second_continuous=None,
):
    """
    Get the value function for a given state and choice vector.

    Args:
        endog_grid_solved (jnp.ndarray): Endogenous wealth grid for all states
            and choices.
        value_solved (jnp.ndarray): Value array for all states and choices.
        params (dict): Dictionary containing model parameters.
        model (dict): Dictionary containing model information and settings.
        state_choice_vec (dict): Dictionary containing a single state and choice.
        wealth (float): The wealth level at which to interpolate.
        second_continuous (float, optional): An additional continuous state
            dimension. If provided, interpolation is done in two dimensions.

    Returns:
        float: The value at the given state and choice.

    """
    map_state_choice_to_index = model["model_structure"][
        "map_state_choice_to_index_with_proxy"
    ]
    discrete_states_names = model["model_structure"]["discrete_states_names"]
    compute_utility = model["model_funcs"]["compute_utility"]

    state_choice_tuple = tuple(
        state_choice_vec[st] for st in discrete_states_names + ["choice"]
    )
    state_choice_index = map_state_choice_to_index[state_choice_tuple]

    if second_continuous is None:
        value = interp_value_on_wealth(
            wealth=wealth,
            endog_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
            value=jnp.take(value_solved, state_choice_index, axis=0),
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
        )
    else:
        value = interp2d_value_on_wealth_and_regular_grid(
            regular_grid=model["model_config"]["continuous_states_info"][
                "second_continuous_grid"
            ],
            wealth_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
            value_grid=jnp.take(value_solved, state_choice_index, axis=0),
            regular_point_to_interp=second_continuous,
            wealth_point_to_interp=wealth,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
        )
    return value


def policy_for_state_choice_vec(
    endog_grid_solved,
    policy_solved,
    model,
    state_choice_vec,
    wealth,
    second_continuous=None,
):
    """Get the policy function for a given state and choice vector.

    Args:
        endog_grid_solved (jnp.ndarray): Endogenous wealth grid for all states
            and choices.
        policy_solved (jnp.ndarray): Policy array for all states and choices.
        model (dict): Dictionary containing model information and settings.
        state_choice_vec (dict): Dictionary containing a single state and choice.
        wealth (float): The wealth level at which to interpolate.
        second_continuous (float, optional): An additional continuous state
            dimension. If provided, interpolation is done in two dimensions.

    Returns:
        float: The policy at the given state and choice.

    """
    map_state_choice_to_index = model["model_structure"][
        "map_state_choice_to_index_with_proxy"
    ]
    discrete_states_names = model["model_structure"]["discrete_states_names"]

    state_choice_tuple = tuple(
        state_choice_vec[st] for st in discrete_states_names + ["choice"]
    )
    state_choice_index = map_state_choice_to_index[state_choice_tuple]

    if second_continuous is None:
        policy = interp_policy_on_wealth(
            wealth=wealth,
            endog_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
            policy=jnp.take(policy_solved, state_choice_index, axis=0),
        )
    else:
        policy = interp2d_policy_on_wealth_and_regular_grid(
            regular_grid=model["model_config"]["continuous_states_info"][
                "second_continuous_grid"
            ],
            wealth_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
            policy_grid=jnp.take(policy_solved, state_choice_index, axis=0),
            regular_point_to_interp=second_continuous,
            wealth_point_to_interp=wealth,
        )

    return policy


def get_state_choice_index_per_discrete_state(
    map_state_choice_to_index, states, discrete_states_names
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


def get_state_choice_index_per_discrete_state_and_choice(model, state_choice_dict):
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
    map_state_choice_to_index = model["model_structure"][
        "map_state_choice_to_index_with_proxy"
    ]
    discrete_states_names = model["model_structure"]["discrete_states_names"]
    state_choice_tuple = tuple(
        state_choice_dict[st] for st in discrete_states_names + ["choice"]
    )
    state_choice_index = map_state_choice_to_index[state_choice_tuple]

    return state_choice_index


def validate_exogenous_processes(model, params):
    """Validate the exogenous processes in the model.

    This function checks that transition probabilities for each exogenous
    process are non-negative, sum to 1, and have the correct dimensionality.
    It raises a ValueError if any checks fail.

    Args:
        model (dict): A dictionary representing the model. Must contain
            'model_funcs' with 'processed_exog_funcs', 'model_structure'
            with 'state_choice_space_dict', and relevant 'options' keys.
        params (dict): Dictionary containing the model parameters.

    Returns:
        bool: True if all exogenous processes are valid; otherwise, a
        ValueError is raised.

    """
    # Update to float64
    jax.config.update("jax_enable_x64", True)

    processed_exog_funcs = model["model_funcs"]["processed_exog_funcs"]
    state_choice_space_dict = model["model_structure"]["state_choice_space_dict"]

    for exog_name, exog_func in processed_exog_funcs.items():
        # Sum transition probabilities for each state-choice combination
        all_transitions = jax.vmap(exog_vec, in_axes=(0, None, None))(
            state_choice_space_dict, exog_func, params
        )
        summed_transitions = jnp.sum(all_transitions, axis=1)

        # Check dtype
        if summed_transitions.dtype != jnp.float64:
            raise ValueError(
                f"Exogenous process {exog_name} does not return float "
                f"transition probabilities. Got {summed_transitions.dtype}"
            )

        # Check non-negativity
        if not (all_transitions >= 0).all():
            raise ValueError(
                f"Exogenous process {exog_name} returns one or more negative "
                f"transition probabilities. An example state choice "
                f"combination is \n\n{pd.Series(state_choice_space_dict).iloc[0]}"
                f"\n\nwith transitions {all_transitions[0]}"
            )

        # Check <= 1
        if not (all_transitions <= 1).all():
            raise ValueError(
                f"Exogenous process {exog_name} returns one or more transition "
                f"probabilities > 1. An example state choice combination is "
                f"\n\n{pd.Series(state_choice_space_dict).iloc[0]}"
                f"\n\nwith transitions {all_transitions[0]}"
            )

        # Check the number of transitions
        n_states = len(model["model_config"]["exogenous_processes"][exog_name])
        if all_transitions.shape[1] != n_states:
            raise ValueError(
                f"Exogenous process {exog_name} does not return the correct "
                f"number of transitions. Expected {n_states}, got "
                f"{all_transitions.shape[1]}."
            )

        # Check sum to 1
        bool_equal_1 = jnp.isclose(
            summed_transitions, jnp.ones_like(summed_transitions)
        )
        if not bool_equal_1.all():
            not_true = jnp.where(~bool_equal_1)[0][0]
            example_state_choice = {
                key: int(value[not_true])
                for key, value in state_choice_space_dict.items()
            }
            raise ValueError(
                f"Exogenous process {exog_name} transition probabilities "
                f"do not sum to 1. An example state choice combination is "
                f"\n\n{pd.Series(example_state_choice)}"
                f"\n\nwith summed transitions {summed_transitions[not_true]}"
                f"\n\nand transitions {all_transitions[not_true]}"
            )

    return True


def exog_vec(state_choice_vec_dict, exog_func, params):
    """
    Evaluate the exogenous function for a given state-choice vector and params.

    Args:
        state_choice_vec_dict (dict): Dictionary containing state-choice values.
        exog_func (Callable): Exogenous process function to be evaluated.
        params (dict): Dictionary of model parameters.

    Returns:
        jnp.ndarray or float: The exogenous process outcomes for the given
        state-choice combination and parameters.
    """
    return exog_func(**state_choice_vec_dict, params=params)
