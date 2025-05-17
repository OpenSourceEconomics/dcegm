from functools import partial
from typing import Callable

import numpy as np
from jax import numpy as jnp

from dcegm.pre_processing.model_structure.shared import span_subspace
from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)


def create_exog_transition_function(model_config, model_specs, continuous_state_name):
    """Create the exogenous process transition function.

    The output function takes a state-choice vector, params and model_specs as input.
    It creates a transition vector over cartesian product of exogenous states.

    """
    if "exogenous_processes" not in model_config:
        options["state_space"]["exogenous_states"] = {"exog_state": [0]}
        compute_exog_transition_vec = return_dummy_exog_transition
        processed_exog_funcs_dict = {}
    else:
        exog_funcs, processed_exog_funcs_dict = process_exog_funcs(
            options, continuous_state_name
        )

        compute_exog_transition_vec = partial(
            get_exog_transition_vec, exog_funcs=exog_funcs
        )
    return compute_exog_transition_vec, processed_exog_funcs_dict


def process_exog_funcs(modeL_config, model_specs, continuous_state_name):
    """Process exogenous functions.

    Args:
        options (dict): Options dictionary.

    Returns:
        tuple: Tuple of exogenous processes.

    """
    exog_processes = options["state_space"]["exogenous_processes"]

    exog_funcs = []
    processed_exog_funcs = {}

    # What about vectors instead of callables supplied?
    for exog_name, exog_dict in exog_processes.items():
        if isinstance(exog_dict["transition"], Callable):
            processed_exog_func = determine_function_arguments_and_partial_model_specs(
                func=exog_dict["transition"],
                model_specs=options["model_params"],
                continuous_state_name=continuous_state_name,
            )
            exog_funcs += [processed_exog_func]
            processed_exog_funcs[exog_name] = processed_exog_func
    return exog_funcs, processed_exog_funcs


def get_exog_transition_vec(exog_funcs, params, **state_choice_vars):
    trans_vector = exog_funcs[0](**state_choice_vars, params=params)

    for exog_func in exog_funcs[1:]:
        # options already partialled in
        trans_vector = jnp.kron(
            trans_vector, exog_func(**state_choice_vars, params=params)
        )

    return trans_vector


def return_dummy_exog_transition(*args, **kwargs):
    return jnp.array([1])


def create_exog_state_mapping(exog_state_space, exog_names):
    def exog_state_mapping(exog_proc_state):
        # Caution: JAX does not throw an error if the exog_proc_state is out of bounds
        # If the index is out of bounds, the last element of the array is returned.
        exog_state = jnp.take(exog_state_space, exog_proc_state, axis=0)
        exog_state_dict = {
            key: jnp.take(exog_state, i) for i, key in enumerate(exog_names)
        }
        return exog_state_dict

    return exog_state_mapping


def process_exog_model_specifications(state_space_options):
    if "exogenous_processes" in state_space_options:
        exog_state_names = list(state_space_options["exogenous_processes"].keys())
        dict_of_only_states = {
            key: state_space_options["exogenous_processes"][key]["states"]
            for key in exog_state_names
        }

        exog_state_space = span_subspace(
            subdict_of_space=dict_of_only_states,
            states_names=exog_state_names,
        )
    else:
        exog_state_names = ["dummy_exog"]
        exog_state_space = np.array([[0]], dtype=np.uint8)

    return exog_state_names, exog_state_space
