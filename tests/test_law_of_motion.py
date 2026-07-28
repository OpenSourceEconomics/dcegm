from itertools import product
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap
from numpy.testing import assert_array_almost_equal as aaae
from scipy.special import roots_sh_legendre
from scipy.stats import norm

import dcegm
import dcegm.toy_models as toy_models
from dcegm.law_of_motion import (
    calc_cont_grids_next_period,
    calc_law_of_motion_for_state_choices,
    calculate_continuous_state,
)
from dcegm.pre_processing.check_params import process_params
from dcegm.toy_models.cons_ret_model_dcegm_paper import budget_constraint

# =====================================================================================
# Auxiliary functions
# =====================================================================================


@jax.jit
def budget_constraint_based_on_experience(
    period: int,
    lagged_choice: int,
    continuous_state_beginning_of_period: float,
    asset_end_of_previous_period: float,
    income_shock_previous_period: float,
    params: Dict[str, float],
) -> float:

    experience_years = continuous_state_beginning_of_period * period

    wage = _calc_stochastic_income_for_experience(
        experience=experience_years,
        lagged_choice=lagged_choice,
        wage_shock=income_shock_previous_period,
        params=params,
    )
    working_hours = _transform_lagged_choice_to_working_hours(lagged_choice)

    wealth_beginning_of_period = (
        wage * working_hours * (lagged_choice > 0)
        + (1 + params["interest_rate"]) * asset_end_of_previous_period
    )

    return jnp.maximum(wealth_beginning_of_period, params["consumption_floor"])


def _calc_stochastic_income_for_experience(
    experience: float,
    lagged_choice: float,
    wage_shock: float,
    params: Dict[str, float],
) -> float:
    """Computes the current level of deterministic and stochastic income."""

    log_wage = (
        params["constant"]
        + params["exp"] * experience
        + params["exp_squared"] * experience**2
        + params["part_time"] * (lagged_choice == 1)
    )

    return jnp.exp(log_wage + wage_shock)


def _transform_lagged_choice_to_working_hours(lagged_choice):

    not_working = lagged_choice == 0
    part_time = lagged_choice == 1
    full_time = lagged_choice == 2

    return not_working * 0 + part_time * 2000 + full_time * 3000


def _next_period_continuous_state(period, lagged_choice, continuous_state, params):

    working_hours = _transform_lagged_choice_to_working_hours(lagged_choice)

    return 1 / (period + 1) * (period * continuous_state + (working_hours) / 3000)


# =====================================================================================
# Tests
# =====================================================================================


model = ["deaton", "retirement_with_shocks", "retirement_no_shocks"]
period = [0, 5, 7]
labor_choice = [0, 1]
max_wealth = [11, 33, 50]
n_grid_points = [101, 444, 1000]

TEST_CASES = list(product(model, period, labor_choice, max_wealth, n_grid_points))


@pytest.mark.parametrize(
    "model_name, period, labor_choice, max_wealth, n_grid_points", TEST_CASES
)
def test_get_beginning_of_period_wealth(
    model_name,
    period,
    labor_choice,
    max_wealth,
    n_grid_points,
):

    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_" + model_name
        )
    )

    params["part_time"] = -1

    n_quad_points = model_config["n_quad_points"]

    income_shock_std = params["income_shock_std"]
    r = params["interest_rate"]
    consump_floor = params["consumption_floor"]

    child_state_dict = {"period": period, "lagged_choice": labor_choice}
    savings_grid = np.linspace(0, max_wealth, n_grid_points)

    _quad_points, _ = roots_sh_legendre(n_quad_points)
    quad_points = norm.ppf(_quad_points) * income_shock_std

    random_saving_scalar = np.random.randint(0, n_grid_points)
    random_shock_scalar = np.random.randint(0, n_quad_points)

    wealth_beginning_of_period = budget_constraint(
        **child_state_dict,
        asset_end_of_previous_period=savings_grid[random_saving_scalar],
        income_shock_previous_period=quad_points[random_shock_scalar],
        model_specs=model_specs,
        params=params,
    )

    if labor_choice == 0:
        age = model_specs["min_age"] + period
        exp_income = (
            params["constant"] + params["exp"] * age + params["exp_squared"] * age**2
        )
        labor_income = jnp.exp(exp_income + quad_points[random_shock_scalar])
    elif labor_choice == 1:
        labor_income = 0
    else:
        raise ValueError("Labor choice not defined")

    budget_expected = (1 + r) * savings_grid[random_saving_scalar] + labor_income

    aaae(wealth_beginning_of_period, max(consump_floor, budget_expected))


TEST_CASES_SECOND_CONTINUOUS = list(product(model, max_wealth, n_grid_points))


@pytest.mark.parametrize(
    "model_name, max_wealth, n_grid_points", TEST_CASES_SECOND_CONTINUOUS
)
def test_wealth_and_second_continuous_state(model_name, max_wealth, n_grid_points):

    # parametrize over number of experience points
    n_exp_points = 10

    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_" + model_name
        )
    )

    model_specs["working_hours_max"] = 3000
    params["part_time"] = -1

    experience_grid = np.linspace(0, 1, n_exp_points)

    child_state_dict = {
        "period": jnp.array([0, 0, 0, 1, 1, 1]),
        "lagged_choice": jnp.array([0, 1, 2, 0, 1, 2]),
    }

    update_experience_vectorized = vmap(
        lambda period, lagged_choice: _next_period_continuous_state(
            period, lagged_choice, experience_grid, params
        )
    )
    experience_next = update_experience_vectorized(
        child_state_dict["period"], child_state_dict["lagged_choice"]
    )

    exp_next = calculate_continuous_state(
        child_state_dict,
        {"continuous_state": experience_grid},
        params,
        _next_period_continuous_state,
    )

    aaae(exp_next, experience_next)


# =====================================================================================
# On-demand vs. full-state-space law of motion equivalence
# =====================================================================================


def _check_subset_matches_full(model, params):
    """calc_law_of_motion_for_state_choices on a state-choice subset must match
    calc_cont_grids_next_period's full-state-space computation for the same underlying
    states -- and must be identical across different choices sharing the same state
    (confirming the choice-drop / no-dedup behavior is correct)."""
    model_structure = model.model_structure
    model_config = model.model_config
    model_funcs = model.model_funcs
    continuous_states_info = model_config["continuous_states_info"]

    full = calc_cont_grids_next_period(
        params=params,
        income_shock_draws_unscaled=model.income_shock_draws_unscaled,
        model_structure=model_structure,
        model_config=model_config,
        model_funcs=model_funcs,
    )

    state_choice_space_dict = model_structure["state_choice_space_dict"]
    map_state_choice_to_parent_state = model_structure[
        "map_state_choice_to_parent_state"
    ]

    # Pick >=2 state-choice indices sharing the same parent state (different
    # choices, same state) plus a few others, to test the no-dedup property.
    values, counts = np.unique(map_state_choice_to_parent_state, return_counts=True)
    shared_parent_state = values[counts >= 2][0]
    idx_sharing_state = np.where(
        map_state_choice_to_parent_state == shared_parent_state
    )[0][:2]
    other_idx = np.where(map_state_choice_to_parent_state != shared_parent_state)[0][:3]
    test_idx = np.concatenate([idx_sharing_state, other_idx])

    state_choice_subset = {
        key: jnp.asarray(var[test_idx]) for key, var in state_choice_space_dict.items()
    }

    income_shock_std = model_funcs["read_funcs"]["income_shock_std"](params)
    income_shock_mean = model_funcs["read_funcs"]["income_shock_mean"](params)
    income_shocks_scaled = (
        model.income_shock_draws_unscaled * income_shock_std + income_shock_mean
    )

    subset_result = calc_law_of_motion_for_state_choices(
        state_choice_vec=state_choice_subset,
        continuous_state_space=model_structure["continuous_state_space"],
        assets_grid_end_of_period=continuous_states_info["assets_grid_end_of_period"],
        income_shocks_scaled=income_shocks_scaled,
        params=params,
        model_funcs=model_funcs,
        has_additional_continuous_states=continuous_states_info[
            "has_additional_continuous_state"
        ],
    )

    expected_parent_states = map_state_choice_to_parent_state[test_idx]
    expected = full["assets_begin_of_period"][expected_parent_states]
    np.testing.assert_allclose(
        np.asarray(subset_result["assets_begin_of_period"]), np.asarray(expected)
    )
    if continuous_states_info["has_additional_continuous_state"]:
        for key, expected_cont in full["continuous_states"].items():
            np.testing.assert_allclose(
                np.asarray(subset_result["continuous_states"][key]),
                np.asarray(expected_cont[expected_parent_states]),
            )

    # Two indices sharing the same parent state (different choices) must give
    # IDENTICAL wealth transitions -- confirms "choice" has no effect, as intended.
    np.testing.assert_array_equal(
        np.asarray(subset_result["assets_begin_of_period"][0]),
        np.asarray(subset_result["assets_begin_of_period"][1]),
    )


def _build_model(model_name):
    model_funcs = toy_models.load_example_model_functions(model_name)
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(model_name)
    )
    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_funcs,
    )
    return model, params


def test_law_of_motion_subset_matches_full_discrete():
    # Retirement model: >=2 choices per state, no additional continuous state.
    model, params = _build_model("dcegm_paper_retirement_no_shocks")
    _check_subset_matches_full(model, params)


def test_law_of_motion_subset_matches_full_cont_exp():
    # >=2 choices per state, plus an additional continuous state ("experience").
    model, params = _build_model("with_cont_exp")
    _check_subset_matches_full(model, params)
