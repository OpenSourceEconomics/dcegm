import jax.numpy as jnp
import numpy as np
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.pre_processing.state_space import (
    determine_function_arguments_and_partial_options,
)
from dcegm.pre_processing.state_space import process_exog_model_specifications
from numpy.testing import assert_almost_equal as aaae

from tests.two_period_models.model import prob_exog_health


def trans_prob_care_demand(health_state, params):
    prob_care_demand = (
        (health_state == 0) * params["care_demand_good_health"]
        + (health_state == 1) * params["care_demand_medium_health"]
        + (health_state == 2) * params["care_demand_bad_health"]
    )

    return prob_care_demand


def test_exog_processes(
    state_space_functions, params_and_options_exog_ltc_and_job_offer
):
    _params, _options = params_and_options_exog_ltc_and_job_offer

    options = _options.copy()
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


def test_nested_exog_process():
    """Test that nested exogenous transition probs are calculated correctly.

    >>> 0.3 * 0.8 + 0.3 * 0.7 + 0.4 * 0.6
    0.69
    >>> 0.3 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4
    0.31000000000000005

    """
    params = {
        "care_demand_good_health": 0.2,
        "care_demand_medium_health": 0.3,
        "care_demand_bad_health": 0.4,
    }

    trans_probs_health = jnp.array([0.3, 0.3, 0.4])

    prob_care_good = trans_prob_care_demand(health_state=0, params=params)
    prob_care_medium = trans_prob_care_demand(health_state=1, params=params)
    prob_care_bad = trans_prob_care_demand(health_state=2, params=params)

    _trans_probs_care_demand = jnp.array(
        [prob_care_good, prob_care_medium, prob_care_bad]
    )
    joint_trans_prob = trans_probs_health @ _trans_probs_care_demand
    expected = 0.3 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4

    aaae(joint_trans_prob, expected)
