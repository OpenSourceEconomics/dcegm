from functools import partial
from typing import Callable

from jax import numpy as jnp

from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options


def create_exog_transition_function(options):
    """Create the exogenous process transition function.

    The output function takes a state vector(also choice?), params and options as input.
    It creates a transition vector over cartesian product of exogenous states.

    """
    if "exogenous_processes" not in options["state_space"]:
        options["state_space"]["exogenous_states"] = {"exog_state": [0]}
        compute_exog_transition_vec = return_dummy_exog_transition
    else:
        exog_funcs = process_exog_funcs(options)

        compute_exog_transition_vec = partial(
            get_exog_transition_vec, exog_funcs=exog_funcs
        )
    return compute_exog_transition_vec


def process_exog_funcs(options):
    """Process exogenous functions.

    Args:
        options (dict): Options dictionary.

    Returns:
        tuple: Tuple of exogenous processes.

    """
    exog_processes = options["state_space"]["exogenous_processes"]

    exog_funcs = []

    # What about vectors instead of callables supplied?
    for exog in exog_processes.values():
        if isinstance(exog["transition"], Callable):
            exog_funcs += [
                determine_function_arguments_and_partial_options(
                    func=exog["transition"],
                    options=options["model_params"],
                )
            ]

    return exog_funcs


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
