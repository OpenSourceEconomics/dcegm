from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent


def test_partial_solve_func():
    model_funcs = toy_models.load_example_model_functions("dcegm_paper")

    model_name = "retirement_with_shocks"
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_" + model_name
        )
    )

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_funcs,
    )

    model_solved = model.solve(params)

    partial_sol = model.solve_partially(
        params=params,
        n_periods=model_config["n_periods"],
        return_candidates=True,
    )

    # Now without loop
    aaae(model_solved.policy, partial_sol["policy"])
    aaae(model_solved.value, partial_sol["value"])
    aaae(model_solved.endog_grid, partial_sol["endog_grid"])

    state_choices = model_solved.model_structure["state_choice_space"]
    choices = state_choices[:, -1]
    states_dict = {
        state: state_choices[:, id]
        for id, state in enumerate(
            model_solved.model_structure["discrete_states_names"]
        )
    }
    states_dict["assets_begin_of_period"] = model_solved.endog_grid[:, 5]
    value_states_all_choices = model_solved.choice_values_for_states(states=states_dict)

    # Take in each row the value corresponding to the choice made
    value_choices = value_states_all_choices[
        np.arange(value_states_all_choices.shape[0]), choices
    ]

    aaae(model_solved.value[:, 5], value_choices)

    # Same for policies
    policy_states_all_choices = model_solved.choice_policies_for_states(
        states=states_dict
    )
    policy_choices = policy_states_all_choices[
        np.arange(policy_states_all_choices.shape[0]), choices
    ]
    aaae(model_solved.policy[:, 5], policy_choices)
