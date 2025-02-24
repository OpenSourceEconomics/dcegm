from functools import partial

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
    second_continous=None,
):
    """Get policy and value for a given state and choice vector.

    Args:
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        model (Model): Model object.
        params (Dict): Dictionary containing the model parameters.

    Returns:
        Tuple[float, float]: Policy and value for the given state and choice vector.

    """
    map_state_choice_to_index = model["model_structure"][
        "map_state_choice_to_index_with_proxy"
    ]
    discrete_states_names = model["model_structure"]["discrete_states_names"]

    state_choice_tuple = tuple(
        state_choice_vec[st] for st in discrete_states_names + ["choice"]
    )

    state_choice_index = map_state_choice_to_index[state_choice_tuple]

    if second_continous is None:
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
            regular_grid=model["options"]["exog_grids"]["second_continuous"],
            wealth_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
            value_grid=jnp.take(value_solved, state_choice_index, axis=0),
            policy_grid=jnp.take(policy_solved, state_choice_index, axis=0),
            regular_point_to_interp=second_continous,
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
    second_continous=None,
):
    """Get policy and value for a given state and choice vector.

    Args:
        state_choice_vec (Dict): Dictionary containing a single state and choice.
        model (Model): Model object.
        params (Dict): Dictionary containing the model parameters.

    Returns:
        Tuple[float, float]: Policy and value for the given state and choice vector.

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

    if second_continous is None:
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
            regular_grid=model["options"]["exog_grids"]["second_continuous"],
            wealth_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
            value_grid=jnp.take(value_solved, state_choice_index, axis=0),
            regular_point_to_interp=second_continous,
            wealth_point_to_interp=wealth,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
        )
    return value


def policy_for_state_choice_vec(
    state_choice_vec,
    wealth,
    map_state_choice_to_index,
    discrete_states_names,
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
        state_choice_vec[st] for st in discrete_states_names + ["choice"]
    )

    state_choice_index = map_state_choice_to_index[state_choice_tuple]

    policy = interp_policy_on_wealth(
        wealth=wealth,
        endog_grid=jnp.take(endog_grid_solved, state_choice_index, axis=0),
        policy=jnp.take(policy_solved, state_choice_index, axis=0),
    )

    return policy


def get_state_choice_index_per_discrete_state(
    map_state_choice_to_index, states, discrete_states_names
):
    indexes = map_state_choice_to_index[
        tuple((states[key],) for key in discrete_states_names)
    ]
    # As the code above generates a dummy dimension in the first we eliminate that
    return indexes[0]


def validate_exogenous_processes(model, params):
    """Validate exogenous processes.

    Args:
        model
        params

    Returns:
        Tuple[bool, Dict]: Tuple with a boolean indicating if all exogenous processes are valid and a dictionary with the results for each exogenous process

    """

    processed_exog_funcs = model["model_funcs"]["processed_exog_funcs"]
    state_choice_space_dict = model["model_structure"]["state_choice_space_dict"]

    results = {}
    for exog_name, exog_func in processed_exog_funcs.items():

        all_transitions = jax.vmap(exoc_vec, in_axes=(0, None, None))(
            state_choice_space_dict, exog_func, params
        )
        summed_transitions = jnp.sum(all_transitions, axis=1)

        if not jnp.allclose(summed_transitions, jnp.ones_like(summed_transitions)):

            print(
                "transition probabilities for exogenous process: ",
                exog_name,
                " are invalid",
            )

    return True


def exoc_vec(state_choice_vec_dict, exog_func, params):
    return exog_func(**state_choice_vec_dict, params=params)
