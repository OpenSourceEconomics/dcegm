import pickle
from typing import Callable, Dict

import jax

from dcegm.pre_processing.batches.batch_creation import create_batches_and_information
from dcegm.pre_processing.check_model_config import check_model_config_and_process
from dcegm.pre_processing.check_model_specs import extract_model_specs_info
from dcegm.pre_processing.model_functions.process_model_functions import (
    process_model_functions_and_extract_info,
    process_sparsity_condition,
)
from dcegm.pre_processing.model_structure.model_structure import create_model_structure
from dcegm.pre_processing.model_structure.state_space import create_state_space
from dcegm.pre_processing.model_structure.stochastic_states import (
    create_stochastic_state_mapping,
)
from dcegm.pre_processing.shared import create_array_with_smallest_int_dtype


def create_model_dict(
    model_config: Dict,
    model_specs: Dict,
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable] = None,
    stochastic_states_transitions: Dict[str, Callable] = None,
    shock_functions: Dict[str, Callable] = None,
    debug_info: str = None,
):
    """Set up the model for dcegm.

    It consists of two steps. First it processes the user supplied functions to make
    them compatible with the interface the dcegm software expects. Second it creates
    the states and choice objects used by the dcegm software.

    Args:
        options (Dict[str, int]): Options dictionary.
        state_space_functions (Dict[str, Callable]): Dictionary of user supplied
        functions for computation of:
            (i) next period endogenous states
            (ii) next period exogenous states
            (iii) next period discrete choices
        utility_functions (Dict[str, Callable]): Dictionary of three user-supplied
            functions for computation of:
            (i) utility
            (ii) inverse marginal utility
            (iii) next period marginal utility
        utility_functions_final_period (Dict[str, Callable]): Dictionary of two
            user-supplied functions for computation of:
            (i) utility
            (ii) next period marginal utility
        budget_constraint (Callable): User supplied budget constraint.

    """
    if debug_info is not None:
        debug_dict = process_debug_string(
            debug_output=debug_info,
            state_space_functions=state_space_functions,
            model_specs=model_specs,
            model_config=model_config,
        )
        if debug_dict["return_output"]:
            return debug_dict["debug_output"]

    model_config_processed = check_model_config_and_process(model_config)

    model_funcs, model_config_processed = process_model_functions_and_extract_info(
        model_config=model_config_processed,
        model_specs=model_specs,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
        stochastic_states_transitions=stochastic_states_transitions,
        shock_functions=shock_functions,
    )

    specs_read_funcs, specs_params_info = extract_model_specs_info(model_specs)
    model_funcs["read_funcs"] = specs_read_funcs

    model_config_processed["params_check_info"] = {
        **model_config_processed["params_check_info"],
        **specs_params_info,
    }

    model_structure = create_model_structure(
        model_config=model_config_processed,
        model_funcs=model_funcs,
    )

    model_funcs["stochastic_state_mapping"] = create_stochastic_state_mapping(
        model_structure["stochastic_state_space"],
        model_structure["stochastic_states_names"],
    )

    print("State, state-choice and child state mapping created.\n")
    print("Start creating batches for the model.")

    batch_info = create_batches_and_information(
        model_structure=model_structure,
        n_periods=model_config_processed["n_periods"],
        min_period_batch_segments=model_config_processed["min_period_batch_segments"],
    )
    if not debug_info == "all":
        # Delete large arrays which is not needed. Not if all is requested
        # by the debug string.
        model_structure.pop("map_state_choice_to_child_states")
        model_structure.pop("map_state_choice_to_index")

    print("Model setup complete.\n")
    return {
        "model_config": model_config_processed,
        "model_funcs": model_funcs,
        "model_structure": model_structure,
        "batch_info": jax.tree.map(create_array_with_smallest_int_dtype, batch_info),
    }


def create_model_dict_and_save(
    model_config: Dict,
    model_specs: Dict,
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable] = None,
    stochastic_states_transitions: Dict[str, Callable] = None,
    shock_functions: Dict[str, Callable] = None,
    path: str = "model.pkl",
    debug_info=None,
):
    """Set up the model and save.

    Model creation is time-consuming. This function creates the model and saves it to
    file. This way the model can be loaded from file in the future, which is much faster
    than recreating the model from scratch.

    """

    model_dict = create_model_dict(
        model_config=model_config,
        model_specs=model_specs,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
        state_space_functions=state_space_functions,
        stochastic_states_transitions=stochastic_states_transitions,
        shock_functions=shock_functions,
        debug_info=debug_info,
    )

    dict_to_save = {
        "model_structure": model_dict["model_structure"],
        "batch_info": model_dict["batch_info"],
    }
    pickle.dump(dict_to_save, open(path, "wb"))

    return model_dict


def load_model_dict(
    model_config: Dict,
    model_specs: Dict,
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable] = None,
    stochastic_states_transitions: Dict[str, Callable] = None,
    shock_functions: Dict[str, Callable] = None,
    path: str = "model.pkl",
):
    """Load the model from file."""

    model = pickle.load(open(path, "rb"))

    model["model_config"] = check_model_config_and_process(model_config)

    model["model_funcs"], model["model_config"] = (
        process_model_functions_and_extract_info(
            model_config=model["model_config"],
            model_specs=model_specs,
            state_space_functions=state_space_functions,
            utility_functions=utility_functions,
            utility_functions_final_period=utility_functions_final_period,
            budget_constraint=budget_constraint,
            stochastic_states_transitions=stochastic_states_transitions,
            shock_functions=shock_functions,
        )
    )

    specs_read_funcs, specs_params_info = extract_model_specs_info(model_specs)
    model["model_funcs"]["read_funcs"] = specs_read_funcs

    model["model_config"]["params_check_info"] = {
        **model["model_config"]["params_check_info"],
        **specs_params_info,
    }

    model["model_funcs"]["stochastic_state_mapping"] = create_stochastic_state_mapping(
        stochastic_state_space=model["model_structure"]["stochastic_state_space"],
        stochastic_state_names=model["model_structure"]["stochastic_states_names"],
    )

    return model


def process_debug_string(
    debug_output, state_space_functions, model_specs, model_config
):
    if debug_output == "state_space_df":
        sparsity_condition = process_sparsity_condition(
            state_space_functions=state_space_functions, model_specs=model_specs
        )
        out = create_state_space(model_config, sparsity_condition, debugging=True)
        debug_info = {"debug_output": out, "return_output": True}
        return debug_info
    elif debug_output == "all":
        debug_info = {"return_output": False}
        return debug_info

    else:
        raise ValueError("The requested debug output is not implemented.")
