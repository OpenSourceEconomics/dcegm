from functools import partial
from typing import Callable

import numpy as np
from jax import numpy as jnp

from dcegm.pre_processing.model_structure.shared import span_subspace
from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)


def create_stochastic_transition_function(
    stochastic_states_transitions, model_config, model_specs, continuous_state_name
):
    """Create the stochastic process transition function.

    The output function takes a state-choice vector, params and model_specs as input. It
    creates a transition vector over cartesian product of exogenous states.

    """
    if "stochastic_states" not in model_config:
        model_config["stochastic_states"] = {"dummy_stochastic": [0]}
        compute_stochastic_transition_vec = return_dummy_stochastic_transition
        func_dict = {}
    else:
        func_list, func_dict = process_stochastic_transitions(
            stochastic_states_transitions,
            model_config=model_config,
            model_specs=model_specs,
            continuous_state_name=continuous_state_name,
        )

        compute_stochastic_transition_vec = partial(
            get_stochastic_transition_vec, transition_funcs=func_list
        )

    return compute_stochastic_transition_vec, func_dict


def process_stochastic_transitions(
    stochastic_states_transitions, model_config, model_specs, continuous_state_name
):
    """Process stochastic functions.

    Args:
        options (dict): Options dictionary.

    Returns:
        tuple: Tuple of exogenous processes.

    """

    func_list = []
    func_dict = {}

    # What about vectors instead of callables supplied?
    for name in model_config["stochastic_states"].keys():
        func = stochastic_states_transitions[name]
        if isinstance(func, Callable):
            processed_exog_func = determine_function_arguments_and_partial_model_specs(
                func=func,
                model_specs=model_specs,
                continuous_state_name=continuous_state_name,
            )
            func_list += [processed_exog_func]
            func_dict[name] = processed_exog_func
        else:
            raise ValueError(f"Stochastic transition function {name} is not callable. ")

    return func_list, func_dict


def get_stochastic_transition_vec(transition_funcs, params, **state_choice_vars):
    """Return Kron product of stochastic transition functions."""
    trans_vector = transition_funcs[0](**state_choice_vars, params=params)

    for func in transition_funcs[1:]:
        # options already partialled in
        trans_vector = jnp.kron(trans_vector, func(**state_choice_vars, params=params))

    return trans_vector


def return_dummy_stochastic_transition(*args, **kwargs):
    return jnp.array([1])


def create_stochastic_state_mapping(stochastic_state_space, stochastic_state_names):
    def stochastic_state_mapping(state_idx):
        # Caution: JAX does not throw an error if the state_idx is out of bounds
        # If the index is out of bounds, the last element of the array is returned.
        stochastic_state = jnp.take(stochastic_state_space, state_idx, axis=0)
        stochastic_states_dict = {
            key: jnp.take(stochastic_state, i)
            for i, key in enumerate(stochastic_state_names)
        }
        return stochastic_states_dict

    return stochastic_state_mapping


def process_stochastic_model_specifications(model_config):
    if "stochastic_states" in model_config:
        stochastic_state_names = list(model_config["stochastic_states"].keys())
        dict_of_only_states = {
            key: model_config["stochastic_states"][key]
            for key in stochastic_state_names
        }

        stochastic_state_space = span_subspace(
            subdict_of_space=dict_of_only_states,
            states_names=stochastic_state_names,
        )
    else:
        stochastic_state_names = ["dummy_stochastic"]
        stochastic_state_space = np.array([[0]], dtype=np.uint8)

    return stochastic_state_names, stochastic_state_space
