"""Regression test: budget equation returning (wealth, aux) together with a second
continuous state (has_additional_continuous_state=True).

final_periods.calc_value_and_budget_for_each_gridpoint calls
compute_assets_begin_of_period(...) directly and assigns the result to
wealth_final_period without unpacking the (wealth, aux) tuple, unlike
law_of_motion.calc_beginning_of_period_assets_for_single_state, which unpacks
it via check_budget_equation_and_return_wealth_plus_optional_aux. This is only
reached when the model has an additional continuous state (e.g. experience),
which routes the last-period solve through
calc_value_and_budget_for_each_gridpoint instead of
calc_value_and_marg_util_for_each_gridpoint.

Neither existing regression test covers this combination:
- test_sim_with_aux.py exercises the aux tuple, but its model only has a
  single continuous state (assets_end_of_period).
- test_two_period_continuous_experience.py exercises a second continuous
  state, but its budget functions return a bare wealth array (no aux).

The two models below are economically identical (same budget equation), one
returning aux and one not. They must produce identical solved value/policy/
endog_grid arrays.

"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models

EXPERIENCE_GRID_POINTS = 6


def _wealth_beginning_of_period(
    period,
    asset_end_of_previous_period,
    lagged_choice,
    experience,
    income_shock_previous_period,
    params,
):
    working = lagged_choice == 0
    experience_years = experience * period

    labor_income = (
        params["constant"]
        + params["exp"] * experience_years
        + params["exp_squared"] * experience_years**2
    )
    income_from_previous_period = jnp.exp(labor_income + income_shock_previous_period)

    wealth_beginning_of_period = (
        income_from_previous_period * working
        + (1 + params["interest_rate"]) * asset_end_of_previous_period
    )
    return jnp.maximum(wealth_beginning_of_period, params["consumption_floor"])


def budget_constraint_without_aux(
    period,
    asset_end_of_previous_period,
    lagged_choice,
    experience,
    income_shock_previous_period,
    params,
):
    return _wealth_beginning_of_period(
        period,
        asset_end_of_previous_period,
        lagged_choice,
        experience,
        income_shock_previous_period,
        params,
    )


def budget_constraint_with_aux(
    period,
    asset_end_of_previous_period,
    lagged_choice,
    experience,
    income_shock_previous_period,
    params,
):
    wealth = _wealth_beginning_of_period(
        period,
        asset_end_of_previous_period,
        lagged_choice,
        experience,
        income_shock_previous_period,
        params,
    )
    aux_dict = {"income_shock": income_shock_previous_period}
    return wealth, aux_dict


def next_period_experience(period, lagged_choice, experience, params):
    return (1 / period) * ((period - 1) * experience + (lagged_choice == 0))


def next_period_continuous_state(period, lagged_choice, experience, params):
    return {
        "experience": next_period_experience(
            period=period,
            lagged_choice=lagged_choice,
            experience=experience,
            params=params,
        )
    }


def _solve_with_budget_constraint(budget_constraint):
    params = {
        "discount_factor": 0.95,
        "delta": 0.35,
        "rho": 1.95,
        "interest_rate": 0.04,
        "taste_shock_scale": 1,
        "income_shock_std": 1,
        "income_shock_mean": 0,
        "constant": 0.75,
        "exp": 0.04,
        "exp_squared": -0.0002,
        "consumption_floor": 0.001,
    }

    model_specs = {
        "n_periods": 2,
        "n_discrete_choices": 2,
    }

    model_config = {
        "n_periods": 2,
        "choices": np.arange(2),
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(0, 50, 100),
            "experience": jnp.linspace(0, 1, EXPERIENCE_GRID_POINTS),
        },
        "n_quad_points": 5,
    }

    model_functions = toy_models.load_example_model_functions("dcegm_paper")

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions={
            "next_period_continuous_state": next_period_continuous_state,
        },
        utility_functions=model_functions["utility_functions"],
        utility_functions_final_period=model_functions[
            "utility_functions_final_period"
        ],
        budget_constraint=budget_constraint,
    )

    return model.solve(params)


def test_budget_equation_with_aux_and_second_continuous_state():
    """Solving must not fail, and must give the same answer whether or not the budget
    equation returns an aux dict alongside wealth."""
    model_solved_without_aux = _solve_with_budget_constraint(
        budget_constraint_without_aux
    )
    model_solved_with_aux = _solve_with_budget_constraint(budget_constraint_with_aux)

    aaae(model_solved_with_aux.value, model_solved_without_aux.value)
    aaae(model_solved_with_aux.policy, model_solved_without_aux.policy)
    aaae(model_solved_with_aux.endog_grid, model_solved_without_aux.endog_grid)
