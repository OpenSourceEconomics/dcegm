import jax.numpy as jnp

from dcegm.interpolation.interp1d import (
    interp1d_policy_and_value_on_wealth,
    interp_policy_on_wealth,
    interp_value_on_wealth,
)


def policy_and_value_for_state_choice_vec(
    state_choice_vec,
    wealth,
    map_state_choice_to_index,
    state_space_names,
    endog_grid_solved,
    policy_solved,
    value_solved,
    compute_utility,
    params,
):
    """Get policy and value for a given state and choice vector.

    Args:
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        model (Model): Model object.
        params (Dict): Dictionary containing the model parameters.

    Returns:
        Tuple[float, float]: Policy and value for the given state and choice vector.

    """
    state_choice_tuple = tuple(
        state_choice_vec[st] for st in state_space_names + ["choice"]
    )

    state_choice_index = map_state_choice_to_index[state_choice_tuple]
    policy, value = interp1d_policy_and_value_on_wealth(
        wealth=wealth,
        endog_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
        policy=jnp.take(policy_solved, state_choice_index, axis=0),
        value=jnp.take(value_solved, state_choice_index, axis=0),
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
    )
    return policy, value


def value_for_state_choice_vec(
    state_choice_vec,
    wealth,
    map_state_choice_to_index,
    state_space_names,
    endog_grid_solved,
    value_solved,
    compute_utility,
    params,
):
    """Get policy and value for a given state and choice vector.

    Args:
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        model (Model): Model object.
        params (Dict): Dictionary containing the model parameters.

    Returns:
        Tuple[float, float]: Policy and value for the given state and choice vector.

    """
    state_choice_tuple = tuple(
        state_choice_vec[st] for st in state_space_names + ["choice"]
    )

    state_choice_index = map_state_choice_to_index[state_choice_tuple]

    value = interp_value_on_wealth(
        wealth=wealth,
        endog_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
        value=jnp.take(value_solved, state_choice_index, axis=0),
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
    )
    return value


def policy_for_state_choice_vec(
    state_choice_vec,
    wealth,
    map_state_choice_to_index,
    state_space_names,
    endog_grid_solved,
    policy_solved,
):
    """Get policy and value for a given state and choice vector.

    Args:
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        model (Model): Model object.
        params (Dict): Dictionary containing the model parameters.

    Returns:
        Tuple[float, float]: Policy and value for the given state and choice vector.

    """
    state_choice_tuple = tuple(
        state_choice_vec[st] for st in state_space_names + ["choice"]
    )

    state_choice_index = map_state_choice_to_index[state_choice_tuple]

    policy = interp_policy_on_wealth(
        wealth=wealth,
        endog_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
        policy=jnp.take(policy_solved, state_choice_index, axis=0),
    )

    return policy


def get_state_choice_index_per_state(
    map_state_choice_to_index, states, state_space_names
):
    indexes = map_state_choice_to_index[
        tuple((states[key],) for key in state_space_names)
    ]
    # As the code above generates a dummy dimension in the first we eliminate that
    return indexes[0]
