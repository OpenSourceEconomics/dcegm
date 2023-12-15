import numpy as np
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.pre_processing.state_space import (
    determine_function_arguments_and_partial_options,
)
from dcegm.pre_processing.state_space import process_exog_model_specifications

from tests.two_period_models.model_functions import prob_exog_health


def test_exog_processes(
    state_space_functions, params_and_options_exog_ltc_and_job_offer
):
    _params, options = params_and_options_exog_ltc_and_job_offer

    options["state_space"]["exogenous_processes"]["health_mother"] = {
        "transition": prob_exog_health,
        "states": [0, 1, 2],
    }
    options["state_space"]["exogenous_processes"]["health_father"] = {
        "transition": prob_exog_health,
        "states": [0, 1, 2],
    }
    model_params_options = options["model_params"]

    (
        add_exog_state_func,
        exog_states_names,
        num_states_of_all_exog_states,
        n_exog_states,
        exog_state_space,
    ) = process_exog_model_specifications(options["state_space"])

    get_state_specific_choice_set = determine_function_arguments_and_partial_options(
        func=state_space_functions["get_state_specific_choice_set"],
        options=model_params_options,
    )

    update_endog_state_by_state_and_choice = (
        determine_function_arguments_and_partial_options(
            func=state_space_functions["update_endog_state_by_state_and_choice"],
            options=model_params_options,
        )
    )

    *_, exog_state_mapping = create_state_space_and_choice_objects(
        options=options,
        get_state_specific_choice_set=get_state_specific_choice_set,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )
    mother_bad_health = np.where(exog_state_space[:, 2] == 2)[0]

    for exog_state in mother_bad_health:
        assert exog_state_mapping(exog_proc_state=exog_state)["health_mother"] == 2
