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

import dcegm.toy_models as toy_models
from dcegm.law_of_motion import calculate_continuous_state
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
        child_state_dict, experience_grid, params, _next_period_continuous_state
    )

    aaae(exp_next, experience_next)
