from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
from dcegm.interfaces.index_functions import (
    get_child_state_index_per_states_and_choices,
)

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

    partial_sol_2 = model.solve_partially(
        params=params,
        n_periods=model_config["n_periods"],
        return_candidates=False,
    )

    aaae(model_solved.policy, partial_sol_2["policy"])
    aaae(model_solved.value, partial_sol_2["value"])
    aaae(model_solved.endog_grid, partial_sol_2["endog_grid"])

    state_choices = model_solved.model_structure["state_choice_space"]
    choices = state_choices[:, -1]
    states_dict = {
        state: state_choices[:, id]
        for id, state in enumerate(
            model_solved.model_structure["discrete_states_names"]
        )
    }
    states_dict["assets_begin_of_period"] = model_solved.endog_grid[:, 0, 5]
    value_states_all_choices = model_solved.choice_values_for_states(states=states_dict)

    # Take in each row the value corresponding to the choice made
    value_choices = value_states_all_choices[
        np.arange(value_states_all_choices.shape[0]), choices
    ]

    aaae(model_solved.value[:, 0, 5], value_choices)

    # Same for policies
    policy_states_all_choices = model_solved.choice_policies_for_states(
        states=states_dict
    )
    policy_choices = policy_states_all_choices[
        np.arange(policy_states_all_choices.shape[0]), choices
    ]
    aaae(model_solved.policy[:, 0, 5], policy_choices)

    model_solved_fast = model.solve(params)
    aaae(model_solved.value, model_solved_fast.value)
    aaae(model_solved.policy, model_solved_fast.policy)
    aaae(model_solved.endog_grid, model_solved_fast.endog_grid)


def test_get_full_child_states_by_asset_id_and_probs_wealth_only():
    """Since #198, calc_cont_grids_next_period always returns a 4-D wealth array
    (n_states, n_cont, n_assets, n_shocks) with a size-1 dummy continuous dimension for
    wealth-only models, which the debug helper must index."""
    model_funcs = toy_models.load_example_model_functions("dcegm_paper")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_retirement_with_shocks"
        )
    )

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        debug_info="all",
        **model_funcs,
    )

    state = {name: 0 for name in model.model_structure["discrete_states_names"]}
    choice = 0
    asset_id = 5

    child_states_df = model.get_full_child_states_by_asset_id_and_probs(
        state=state,
        choice=choice,
        params=params,
        asset_id=asset_id,
    )

    child_idx = get_child_state_index_per_states_and_choices(
        states=state, choices=choice, model_structure=model.model_structure
    )
    law_of_motions = model.compute_law_of_motions(params=params)
    expected_quad_wealth = law_of_motions["assets_begin_of_period"][
        child_idx, 0, asset_id, :
    ]

    for id_quad in range(expected_quad_wealth.shape[1]):
        aaae(
            child_states_df[f"assets_begin_of_period_quad_point_{id_quad}"].values,
            expected_quad_wealth[:, id_quad],
        )


def test_get_full_child_states_by_asset_id_and_probs_second_continuous():
    """With an additional continuous state, the debug helper must select the child
    states of the requested continuous grid point and report the state under its actual
    name."""
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        debug_info="all",
        **model_funcs,
    )

    state = {name: 0 for name in model.model_structure["discrete_states_names"]}
    choice = 0
    asset_id = 5
    second_continuous_id = 2

    child_states_df = model.get_full_child_states_by_asset_id_and_probs(
        state=state,
        choice=choice,
        params=params,
        asset_id=asset_id,
        second_continuous_id=second_continuous_id,
    )

    child_idx = get_child_state_index_per_states_and_choices(
        states=state, choices=choice, model_structure=model.model_structure
    )
    law_of_motions = model.compute_law_of_motions(params=params)
    expected_quad_wealth = law_of_motions["assets_begin_of_period"][
        child_idx, second_continuous_id, asset_id, :
    ]
    expected_experience = law_of_motions["continuous_states"]["experience"][
        child_idx, second_continuous_id
    ]

    aaae(child_states_df["experience"].values, expected_experience)
    for id_quad in range(expected_quad_wealth.shape[1]):
        aaae(
            child_states_df[f"assets_begin_of_period_quad_point_{id_quad}"].values,
            expected_quad_wealth[:, id_quad],
        )
