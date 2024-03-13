import pickle
from typing import Callable
from typing import Dict

import numpy as np
from dcegm.pre_processing.batches import create_batches_and_information
from dcegm.pre_processing.exog_processes import create_exog_mapping
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects


def setup_model(
    options: Dict,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
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

    (
        model_funcs,
        compute_upper_envelope,
        get_state_specific_choice_set,
        get_next_period_state,
    ) = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    (
        state_space,
        state_space_dict,
        map_state_to_index,
        exog_state_space,
        states_names_without_exog,
        exog_state_names,
        state_choice_space,
        map_state_choice_to_index,
        map_state_choice_vec_to_parent_state,
        map_state_choice_to_child_states,
    ) = create_state_space_and_choice_objects(
        options=options,
        get_state_specific_choice_set=get_state_specific_choice_set,
        get_next_period_state=get_next_period_state,
    )

    exog_mapping = create_exog_mapping(
        exog_state_space.astype(np.int16), exog_state_names
    )

    batch_info = create_batches_and_information(
        state_choice_space=state_choice_space,
        n_periods=options["state_space"]["n_periods"],
        map_state_choice_to_child_states=map_state_choice_to_child_states,
        map_state_choice_to_index=map_state_choice_to_index,
        map_state_choice_vec_to_parent_state=map_state_choice_vec_to_parent_state,
        state_space=state_space,
        state_space_names=states_names_without_exog + exog_state_names,
    )

    model = {
        "model_funcs": model_funcs,
        "compute_upper_envelope": compute_upper_envelope,
        "get_state_specific_choice_set": get_state_specific_choice_set,
        "batch_info": batch_info,
        "state_space": state_space,
        "state_choice_space": state_choice_space,
        "state_space_dict": state_space_dict,
        "state_space_names": states_names_without_exog + exog_state_names,
        "map_state_choice_to_index": map_state_choice_to_index,
        "exog_state_space": exog_state_space,
        "exog_state_names": exog_state_names,
        "exog_mapping": exog_mapping,
        "get_next_period_state": get_next_period_state,
    }
    return model


def setup_and_save_model(
    options: Dict,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    path: str,
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
    )
    array_names = [
        "period_specific_state_objects",
        "state_space",
        "state_space_names",
        "map_state_choice_to_index",
        "exog_state_space",
        "exog_state_names",
        "batch_info",
    ]
    dict_to_save = {key: value for key, value in model.items() if key in array_names}
    pickle.dump(dict_to_save, open(path, "wb"))

    return model


def load_and_setup_model(
    options: Dict,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    path: str,
):
    """Load the model from file."""

    model = pickle.load(open(path, "rb"))
    (
        model["model_funcs"],
        model["compute_upper_envelope"],
        model["get_state_specific_choice_set"],
        model["get_next_period_state"],
    ) = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    model["exog_mapping"] = create_exog_mapping(
        np.array(model["exog_state_space"], dtype=np.int16), model["exog_state_names"]
    )

    return model
