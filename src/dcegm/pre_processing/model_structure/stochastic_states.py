import inspect
from functools import partial
from typing import Callable

import jax
import numpy as np
from jax import numpy as jnp

from dcegm.pre_processing.model_structure.shared import span_subspace
from dcegm.pre_processing.shared import (
    create_array_with_smallest_int_dtype,
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
        func_dict = process_stochastic_transitions(
            stochastic_states_transitions,
            model_config=model_config,
            model_specs=model_specs,
            continuous_state_name=continuous_state_name,
        )

        trans_func_list = [func_dict[name] for name in func_dict.keys()]

        compute_stochastic_transition_vec = partial(
            get_stochastic_transition_vec, transition_funcs=trans_func_list
        )

    return compute_stochastic_transition_vec, func_dict


def process_stochastic_transitions(
    stochastic_states_transitions, model_config, model_specs, continuous_state_name
):
    """Process stochastic functions.

    Args:
        options (dict): Options dictionary.

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

    return func_dict


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


def create_sparse_stochastic_trans_map(
    model_structure, model_funcs, model_config_processed, from_saved=False
):
    """Create sparse mapping from state-choice to stochastic states."""
    state_choice_dict = model_structure["state_choice_space_dict"]
    stochastic_transitions_dict = model_funcs["processed_stochastic_funcs"]
    threshold = 1e-6

    # Add index to state_choice_dict
    n_state_choices = len(state_choice_dict[next(iter(state_choice_dict))])
    sparse_index_functions = []
    spares_stoch_trans_funcs = []
    trans_func_dict = {}

    for stoch_name, stoch_states in model_config_processed["stochastic_states"].items():
        if stoch_name == "dummy_stochastic":
            continue

        has_params = (
            "params"
            in inspect.signature(stochastic_transitions_dict[stoch_name]).parameters
        )
        n_states = len(stoch_states)
        trans_func = stochastic_transitions_dict[stoch_name]

        if has_params:
            index_eval = lambda index, n=n_states: np.ones(n) / n
            sparse_index_functions.append(index_eval)
            spares_stoch_trans_funcs += [trans_func]
            trans_func_dict[stoch_name] = trans_func
        else:

            eval_func = lambda state_choice, f=trans_func: f(**state_choice)

            # Compute transitions and find indices to keep
            single_transitions = jax.vmap(eval_func)(state_choice_dict)
            zero_mask = single_transitions < threshold

            max_n_zeros = zero_mask.sum(axis=1).min()
            if max_n_zeros == 0:
                # No sparsity for this state
                index_eval = lambda index, n=n_states: np.ones(n) / n
                sparse_index_functions.append(index_eval)
                spares_stoch_trans_funcs += [trans_func]
                trans_func_dict[stoch_name] = trans_func
                continue
            n_keep = zero_mask.shape[1] - max_n_zeros

            keep_mask = ~zero_mask
            keep_indices = np.where(keep_mask, np.arange(keep_mask.shape[1]), -1)

            # Get sorted positions and the original indices
            sort_order = np.argsort(keep_indices, axis=1)
            indices_to_keep = np.take_along_axis(keep_indices, sort_order, axis=1)[
                :, -n_keep:
            ]

            # For positions with -1, use the original position from sort_order
            original_positions = sort_order[:, -n_keep:]
            indices_to_keep = np.where(
                indices_to_keep == -1, original_positions, indices_to_keep
            )
            # Sort again to get indices in ascending order
            indices_to_keep = np.sort(indices_to_keep, axis=1)
            indices_to_keep = jnp.asarray(indices_to_keep)
            indices_to_keep = create_array_with_smallest_int_dtype(indices_to_keep)

            def create_sparse_func(trans_f, indices_keep):
                def sparse_trans_func(**kwargs):
                    index_to_keep = indices_keep[kwargs["index"]]
                    return trans_f(**kwargs)[index_to_keep]

                return sparse_trans_func

            sparse_trans_func = create_sparse_func(trans_func, indices_to_keep)

            spares_stoch_trans_funcs += [sparse_trans_func]
            trans_func_dict[stoch_name] = sparse_trans_func

            # Create sparse eval function that returns NaN for deleted states
            def create_nan_padded_eval(indices_keep, n_total):
                def nan_padded_eval(index):
                    result = jnp.full(n_total, jnp.nan)
                    result = result.at[indices_keep[index]].set(1.0)
                    return result

                return nan_padded_eval

            index_eval = create_nan_padded_eval(indices_to_keep, n_states)
            sparse_index_functions.append(index_eval)

    compute_stochastic_transition_vec = partial(
        get_stochastic_transition_vec, transition_funcs=spares_stoch_trans_funcs
    )
    if from_saved:
        return compute_stochastic_transition_vec, trans_func_dict

    # Evaluate kronecker product with NaNs
    def kronecker_with_index(idx):
        trans_vector = sparse_index_functions[0](idx)
        for func in sparse_index_functions[1:]:
            trans_vector = jnp.kron(trans_vector, func(idx))
        return trans_vector

    all_transitions = jax.vmap(kronecker_with_index)(
        jnp.arange(n_state_choices),
    )

    # Find non-NaN positions (states to keep)
    keep_mask = ~np.isnan(all_transitions)

    # Select non-NaN child states directly
    sparse_child_states_mapping = model_structure["map_state_choice_to_child_states"][
        keep_mask
    ].reshape(keep_mask.shape[0], -1)
    state_choice_dict_with_idx = {
        **state_choice_dict,
        "index": jnp.arange(n_state_choices),
    }
    return (
        sparse_child_states_mapping,
        state_choice_dict_with_idx,
        compute_stochastic_transition_vec,
        trans_func_dict,
    )
