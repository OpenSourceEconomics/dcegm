from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
from dcegm.interfaces.index_functions import (
    get_child_state_index_per_states_and_choices,
)
from dcegm.toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent


# ====================================================================================
# Minimal model with two additional continuous states (besides assets), used to
# check that the debug helper generalizes beyond a single additional continuous
# state.
# ====================================================================================


def _next_period_continuous_state_two_cont(lagged_choice, cont_a, cont_b):
    return {
        "cont_a": cont_a + (lagged_choice == 0),
        "cont_b": cont_b + (lagged_choice == 1),
    }


def _budget_constraint_two_cont(
    lagged_choice,
    cont_a,
    cont_b,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    interest_factor = 1 + params["interest_rate"]
    wage = (
        params["wage_constant"] + params["wage_a"] * cont_a + params["wage_b"] * cont_b
    )
    resource = interest_factor * asset_end_of_previous_period + (
        wage + income_shock_previous_period
    ) * (lagged_choice != 0)
    return jnp.maximum(resource, 0.5)


def _setup_two_continuous_states_model(debug_info=None):
    model_specs = {"choices": [0, 1]}
    model_config = {
        "n_periods": 3,
        "choices": [0, 1],
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(0.5, 20, 10),
            "assets_begin_of_period": jnp.linspace(0.5, 20, 10),
            "cont_a": jnp.arange(0, 3, dtype=float),
            "cont_b": jnp.arange(0, 3, dtype=float),
        },
        "n_quad_points": 3,
        "upper_envelope": {"method": "druedahl_jorgensen"},
    }
    state_space_functions = {
        "next_period_continuous_state": _next_period_continuous_state_two_cont,
    }

    return dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        state_space_functions=state_space_functions,
        stochastic_states_transitions={},
        budget_constraint=_budget_constraint_two_cont,
        debug_info=debug_info,
    )


def _params_two_continuous_states():
    return {
        "interest_rate": 0.02,
        "wage_constant": 3.0,
        "wage_a": 0.5,
        "wage_b": 0.8,
        "income_shock_std": 1,
        "income_shock_mean": 0,
        "taste_shock_scale": 1,
        "discount_factor": 0.95,
        "rho": 0.9,
        "delta": 1.5,
    }


def _setup_with_cont_exp_model_and_params():
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
    return model, params


def _setup_two_continuous_states_model_and_params():
    return (
        _setup_two_continuous_states_model(debug_info="all"),
        _params_two_continuous_states(),
    )


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


@pytest.mark.parametrize(
    "setup_model_and_params, continuous_state_names",
    [
        (_setup_with_cont_exp_model_and_params, ["experience"]),
        (_setup_two_continuous_states_model_and_params, ["cont_a", "cont_b"]),
    ],
)
def test_get_full_child_states_by_asset_id_and_probs_second_continuous(
    setup_model_and_params, continuous_state_names
):
    """With additional continuous states, the debug helper must select the child states
    of the requested continuous grid point and report every additional continuous state
    under its actual name.

    Regression test: previously indexed "additional_continuous_state_names"[0] only,
    so with more than one additional continuous state, states beyond the first were
    silently dropped from the returned DataFrame instead of being reported.

    """
    model, params = setup_model_and_params()

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

    for continuous_state_name in continuous_state_names:
        expected_continuous_state = law_of_motions["continuous_states"][
            continuous_state_name
        ][child_idx, second_continuous_id]
        aaae(child_states_df[continuous_state_name].values, expected_continuous_state)

    for id_quad in range(expected_quad_wealth.shape[1]):
        aaae(
            child_states_df[f"assets_begin_of_period_quad_point_{id_quad}"].values,
            expected_quad_wealth[:, id_quad],
        )
