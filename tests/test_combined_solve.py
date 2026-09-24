"""Tests for assembling a big model's solve from smaller sub-model solves.

The model has an invariant "type" deterministic state (never changed by the
deterministic update, so it never crosses a transition). The big model spans both types;
the two sub-models each fix one type. Their state-choice spaces therefore partition the
big model's, and the assembled solve must reproduce the pooled solve.

"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import dcegm

jax.config.update("jax_enable_x64", True)


# ====================================================================================
# Model functions
# ====================================================================================


def flow_util(consumption, choice, type, params):
    rho = params["rho"]
    # Work disutility differs by type, so the two types have different solutions.
    disutility = params["delta"] * (choice == 0) * (1 + 0.5 * type)
    return consumption ** (1 - rho) / (1 - rho) - disutility


def marginal_utility(consumption, params):
    return consumption ** (-params["rho"])


def inverse_marginal_utility(marginal_utility, params):
    return marginal_utility ** (-1 / params["rho"])


def final_period_utility(wealth, choice, params):
    return wealth ** (1 - params["rho"]) / (1 - params["rho"])


def marginal_final(wealth, choice, params):
    return marginal_utility(wealth, params)


def state_specific_choice_set(period, lagged_choice, model_specs):
    # Retirement (choice 1) is absorbing.
    if lagged_choice == 1:
        return np.array([1])
    return np.array([0, 1])


def next_period_deterministic_state(period, choice, lagged_choice):
    # "type" is deliberately not returned -> it is carried over unchanged (invariant).
    return {
        "period": period + 1,
        "lagged_choice": choice,
    }


def sparsity_condition(period, lagged_choice, type):
    return True


def budget_constraint(
    lagged_choice,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    interest_factor = 1 + params["interest_rate"]
    wage = params["wage_constant"] + income_shock_previous_period
    resource = interest_factor * asset_end_of_previous_period + wage * (
        lagged_choice == 0
    )
    return jnp.maximum(resource, 0.5)


utility_functions = {
    "utility": flow_util,
    "inverse_marginal_utility": inverse_marginal_utility,
    "marginal_utility": marginal_utility,
}
utility_functions_final_period = {
    "utility": final_period_utility,
    "marginal_utility": marginal_final,
}
state_space_functions = {
    "state_specific_choice_set": state_specific_choice_set,
    "next_period_deterministic_state": next_period_deterministic_state,
    "sparsity_condition": sparsity_condition,
}

BUILD_KWARGS = dict(
    model_specs={"n_choices": 2},
    utility_functions=utility_functions,
    utility_functions_final_period=utility_functions_final_period,
    state_space_functions=state_space_functions,
    stochastic_states_transitions={},
    budget_constraint=budget_constraint,
)


def _model_config(type_grid):
    return {
        "n_periods": 5,
        "choices": [0, 1],
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(0, 50, 80),
        },
        "deterministic_states": {
            "type": np.asarray(type_grid, dtype=int),
        },
        "n_quad_points": 5,
    }


@pytest.fixture(scope="module")
def params():
    return {
        "interest_rate": 0.03,
        "wage_constant": 2.0,
        "income_shock_std": 1.0,
        "income_shock_mean": 0.0,
        "taste_shock_scale": 1.0,
        "discount_factor": 0.95,
        "rho": 1.5,
        "delta": 0.3,
    }


@pytest.fixture(scope="module")
def big_config():
    return _model_config([0, 1])


@pytest.fixture(scope="module")
def small_models():
    return [
        dcegm.setup_model(model_config=_model_config([0]), **BUILD_KWARGS),
        dcegm.setup_model(model_config=_model_config([1]), **BUILD_KWARGS),
    ]


def test_assembled_solve_matches_pooled(params, big_config, small_models):
    big_model = dcegm.setup_model(model_config=big_config, **BUILD_KWARGS)
    pooled = big_model.solve(params)

    solve_from_small = dcegm.get_solve_from_small_models(
        small_models=small_models,
        parallel=False,
        model_config=big_config,
        **BUILD_KWARGS,
    )
    assembled = solve_from_small(params)

    assert_allclose(np.asarray(assembled.value), np.asarray(pooled.value))
    assert_allclose(np.asarray(assembled.policy), np.asarray(pooled.policy))
    assert_allclose(np.asarray(assembled.endog_grid), np.asarray(pooled.endog_grid))


def test_parallel_matches_pooled(params, big_config, small_models):
    big_model = dcegm.setup_model(model_config=big_config, **BUILD_KWARGS)
    pooled = big_model.solve(params)

    solve_from_small = dcegm.get_solve_from_small_models(
        small_models=small_models,
        parallel=True,
        model_config=big_config,
        **BUILD_KWARGS,
    )
    assembled = solve_from_small(params)

    assert_allclose(np.asarray(assembled.value), np.asarray(pooled.value))
    assert_allclose(np.asarray(assembled.policy), np.asarray(pooled.policy))


def test_overlapping_submodels_raise(big_config):
    # Two identical full-span sub-models cover every row twice -> not a partition.
    overlapping = [
        dcegm.setup_model(model_config=_model_config([0, 1]), **BUILD_KWARGS),
        dcegm.setup_model(model_config=_model_config([0, 1]), **BUILD_KWARGS),
    ]
    with pytest.raises(ValueError, match="partition|sum to"):
        dcegm.get_solve_from_small_models(
            small_models=overlapping,
            parallel=False,
            model_config=big_config,
            **BUILD_KWARGS,
        )


def test_incomplete_submodels_raise(big_config):
    # Only one of the two types -> some big rows are uncovered.
    incomplete = [dcegm.setup_model(model_config=_model_config([0]), **BUILD_KWARGS)]
    with pytest.raises(ValueError, match="partition|sum to"):
        dcegm.get_solve_from_small_models(
            small_models=incomplete,
            parallel=False,
            model_config=big_config,
            **BUILD_KWARGS,
        )


def test_mismatched_continuous_grid_raises(big_config, small_models):
    # A big model with a different wealth grid is inconsistent with the sub-models.
    bad_config = _model_config([0, 1])
    bad_config["continuous_states"]["assets_end_of_period"] = jnp.linspace(0, 50, 79)
    with pytest.raises(ValueError, match="grid|n_total_wealth_grid"):
        dcegm.get_solve_from_small_models(
            small_models=small_models,
            parallel=False,
            model_config=bad_config,
            **BUILD_KWARGS,
        )
