from typing import Callable
from typing import Dict

from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects


def setup_model(
    options: Dict,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
):
    """This function sets up the model used by the dcegm software.

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
        update_endog_state_by_state_and_choice,
    ) = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    (
        period_specific_state_objects,
        state_space,
        state_space_names,
        map_state_choice_to_index,
        exog_mapping,
    ) = create_state_space_and_choice_objects(
        options=options,
        get_state_specific_choice_set=get_state_specific_choice_set,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    model = {
        "model_funcs": model_funcs,
        "compute_upper_envelope": compute_upper_envelope,
        "get_state_specific_choice_set": get_state_specific_choice_set,
        "period_specific_state_objects": period_specific_state_objects,
        "state_space": state_space,
        "state_space_names": state_space_names,
        "map_state_choice_to_index": map_state_choice_to_index,
        "exog_mapping": exog_mapping,
    }
    model[
        "update_endog_state_by_state_and_choice"
    ] = update_endog_state_by_state_and_choice
    return model
