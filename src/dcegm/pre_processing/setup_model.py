import pickle
from typing import Callable, Dict

import jax

from dcegm.pre_processing.batches.batch_creation import create_batches_and_information
from dcegm.pre_processing.check_options import check_options_and_set_defaults
from dcegm.pre_processing.model_functions import (
    process_model_functions,
    process_sparsity_condition,
)
from dcegm.pre_processing.model_structure.exogenous_processes import (
    create_exog_state_mapping,
)
from dcegm.pre_processing.model_structure.model_structure import create_model_structure
from dcegm.pre_processing.model_structure.state_space import create_state_space
from dcegm.pre_processing.shared import create_array_with_smallest_int_dtype


def setup_model(
    options: Dict,
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable] = None,
    debug_info: str = None,
    sim_model=False,
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
        debug_dict = process_debug_string(debug_info, state_space_functions, options)
        if debug_dict["return_output"]:
            return debug_dict["debug_output"]

    options = check_options_and_set_defaults(options)

    model_funcs = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
        sim_model=sim_model,
    )

    model_structure = create_model_structure(
        options=options,
        model_funcs=model_funcs,
    )

    model_funcs["exog_state_mapping"] = create_exog_state_mapping(
        model_structure["exog_state_space"],
        model_structure["exog_states_names"],
    )

    print("State, state-choice and child state mapping created.\n")
    print("Start creating batches for the model.")

    batch_info = create_batches_and_information(
        model_structure=model_structure,
        state_space_options=options["state_space"],
    )
    if not debug_info == "all":
        # Delete large array which is not needed. Not if all is requested
        # by the debug string.
        model_structure.pop("map_state_choice_to_child_states")

    print("Model setup complete.\n")
    return {
        "options": options,
        "model_funcs": model_funcs,
        "model_structure": model_structure,
        "batch_info": jax.tree.map(create_array_with_smallest_int_dtype, batch_info),
    }


def setup_and_save_model(
    options: Dict,
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable] = None,
    path: str = "model.pkl",
    sim_model=False,
):
    """Set up the model and save.

    Model creation is time-consuming. This function creates the model and saves it to
    file. This way the model can be loaded from file in the future, which is much faster
    than recreating the model from scratch.

    """
    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
        sim_model=sim_model,
    )

    dict_to_save = {
        "model_structure": model["model_structure"],
        "batch_info": model["batch_info"],
    }
    pickle.dump(dict_to_save, open(path, "wb"))

    return model


def load_and_setup_model(
    options: Dict,
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable] = None,
    path: str = "model.pkl",
    sim_model=False,
):
    """Load the model from file."""

    model = pickle.load(open(path, "rb"))

    model["options"] = check_options_and_set_defaults(options)

    model["model_funcs"] = process_model_functions(
        options=model["options"],
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
        sim_model=sim_model,
    )

    model["model_funcs"]["exog_state_mapping"] = create_exog_state_mapping(
        exog_state_space=model["model_structure"]["exog_state_space"],
        exog_names=model["model_structure"]["exog_states_names"],
    )

    return model


def process_debug_string(debug_output, state_space_functions, options):
    if debug_output == "state_space_df":
        sparsity_condition = process_sparsity_condition(state_space_functions, options)
        out = create_state_space(
            options["state_space"], sparsity_condition, debugging=True
        )
        debug_info = {"debug_output": out, "return_output": True}
        return debug_info
    elif debug_output == "all":
        debug_info = {"return_output": False}
        return debug_info

    else:
        raise ValueError("The requested debug output is not implemented.")
