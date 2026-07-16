import numpy as np
from numpy.testing import assert_allclose

import dcegm
from dcegm.toy_models.cons_ret_model_dcegm_paper import (
    inverse_marginal_utility_crra,
    marginal_utility_crra,
    marginal_utility_final_consume_all,
)
from tests.test_changing_choice_set import (
    budget,
    choice_set,
    flow_utility,
    next_period_state,
    prob_health,
    prob_partner,
    sparsity_condition,
    utility_final,
)


def _get_model_objects(n_periods):
    params = {
        "rho": 0.5,
        "delta": 1,
        "phi": 0.5,
        "constant": 1,
        "exp": 0.1,
        "exp_squared": -0.01,
        "pension_per_experience": 0.3,
        "unemployment_benefits": 0.4,
        "health_costs": 0.5,
        "consumption_floor": 0,
        "p_bad_health_given_good_health": 0.2,
        "p_bad_health_given_bad_health": 1,
        "p_partner_given_single": 0.5,
        "p_partner_given_partner": 0.9,
    }

    model_specs = {
        "min_age": 0,
        "n_periods": n_periods,
        "n_choices": 3,
        "n_health_states": 2,
        "n_partner_states": 2,
        "max_experience": n_periods - 1,
        "interest_rate": 0.05,
        "discount_factor": 0.95,
        "taste_shock_scale": 1,
        "income_shock_std": 1,
        "income_shock_mean": 0.0,
    }

    model_config = {
        "n_periods": n_periods,
        "choices": np.arange(3),
        "deterministic_states": {
            "experience": np.arange(n_periods),
        },
        "continuous_states": {
            "assets_end_of_period": np.linspace(0, 500, 100),
        },
        "stochastic_states": {
            "health": [0, 1],
            "partner": [0, 1],
        },
        "n_quad_points": 5,
    }

    state_space_functions = {
        "state_specific_choice_set": choice_set,
        "next_period_deterministic_state": next_period_state,
        "sparsity_condition": sparsity_condition,
    }

    utility_functions = {
        "utility": flow_utility,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
    }

    utility_functions_final_period = {
        "utility": utility_final,
        "marginal_utility": marginal_utility_final_consume_all,
    }

    exogenous_states_transition = {
        "health": prob_health,
        "partner": prob_partner,
    }

    return (
        params,
        model_specs,
        model_config,
        state_space_functions,
        utility_functions,
        utility_functions_final_period,
        exogenous_states_transition,
    )


def _solve_with_config(
    model_config,
    model_specs,
    params,
    state_space_functions,
    utility_functions,
    utility_functions_final_period,
    exogenous_states_transition,
):
    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget,
        stochastic_states_transitions=exogenous_states_transition,
    )
    return model.solve(params)


def test_period_max_equals_largest_block_without_segments():
    (
        params,
        model_specs,
        model_config,
        state_space_functions,
        utility_functions,
        utility_functions_final_period,
        exogenous_states_transition,
    ) = _get_model_objects(n_periods=8)

    model_config_baseline = dict(model_config)
    model_config_baseline["batch_mode"] = "largest_block"

    model_config_period_max = dict(model_config)
    model_config_period_max["batch_mode"] = "period_max"

    solved_baseline = _solve_with_config(
        model_config=model_config_baseline,
        model_specs=model_specs,
        params=params,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        exogenous_states_transition=exogenous_states_transition,
    )
    solved_period_max = _solve_with_config(
        model_config=model_config_period_max,
        model_specs=model_specs,
        params=params,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        exogenous_states_transition=exogenous_states_transition,
    )

    assert_allclose(solved_baseline.value, solved_period_max.value, equal_nan=True)
    assert_allclose(solved_baseline.policy, solved_period_max.policy, equal_nan=True)
    assert_allclose(
        solved_baseline.endog_grid, solved_period_max.endog_grid, equal_nan=True
    )


def test_period_max_equals_largest_block_with_segments():
    (
        params,
        model_specs,
        model_config,
        state_space_functions,
        utility_functions,
        utility_functions_final_period,
        exogenous_states_transition,
    ) = _get_model_objects(n_periods=8)

    model_config_baseline = dict(model_config)
    model_config_baseline["min_period_batch_segments"] = [2, 3]
    model_config_baseline["batch_mode"] = [
        "largest_block",
        "largest_block",
        "largest_block",
    ]

    model_config_period_max = dict(model_config)
    model_config_period_max["min_period_batch_segments"] = [2, 3]
    model_config_period_max["batch_mode"] = [
        "period_max",
        "largest_block",
        "period_max",
    ]

    solved_baseline = _solve_with_config(
        model_config=model_config_baseline,
        model_specs=model_specs,
        params=params,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        exogenous_states_transition=exogenous_states_transition,
    )
    solved_period_max = _solve_with_config(
        model_config=model_config_period_max,
        model_specs=model_specs,
        params=params,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        exogenous_states_transition=exogenous_states_transition,
    )

    assert_allclose(solved_baseline.value, solved_period_max.value, equal_nan=True)
    assert_allclose(solved_baseline.policy, solved_period_max.policy, equal_nan=True)
    assert_allclose(
        solved_baseline.endog_grid, solved_period_max.endog_grid, equal_nan=True
    )
