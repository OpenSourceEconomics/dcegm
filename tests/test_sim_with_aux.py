import jax.numpy as jnp
import numpy as np
import pytest

import dcegm
import dcegm.toy_models as toy_models


def budget_with_aux(
    period,
    lagged_choice,
    asset_end_of_previous_period,
    income_shock_previous_period,
    model_specs,
    params,
):
    wealth, shock, income = budget_constraint_raw(
        period,
        lagged_choice,
        asset_end_of_previous_period,
        income_shock_previous_period,
        model_specs,
        params,
    )
    aux_dict = {
        "income": income,
    }
    return wealth, aux_dict


def budget_without_aux(
    period,
    lagged_choice,
    asset_end_of_previous_period,
    income_shock_previous_period,
    model_specs,
    params,
):
    wealth, _, _ = budget_constraint_raw(
        period,
        lagged_choice,
        asset_end_of_previous_period,
        income_shock_previous_period,
        model_specs,
        params,
    )
    return wealth


def budget_constraint_raw(
    period,
    lagged_choice,
    asset_end_of_previous_period,
    income_shock_previous_period,
    model_specs,
    params,
):
    # Calculate stochastic labor income
    income_from_previous_period = _calc_stochastic_income(
        period=period,
        lagged_choice=lagged_choice,
        wage_shock=income_shock_previous_period,
        min_age=model_specs["min_age"],
        constant=params["constant"],
        exp=params["exp"],
        exp_squared=params["exp_squared"],
    )

    wealth_beginning_of_period = (
        income_from_previous_period
        + (1 + params["interest_rate"]) * asset_end_of_previous_period
    )

    # Retirement safety net, only in retirement model, but we require to have it always
    # as a parameter
    wealth_beginning_of_period = jnp.maximum(
        wealth_beginning_of_period, params["consumption_floor"]
    )

    return (
        wealth_beginning_of_period,
        income_shock_previous_period,
        income_from_previous_period,
    )


def _calc_stochastic_income(
    period,
    lagged_choice,
    wage_shock,
    min_age,
    constant,
    exp,
    exp_squared,
):
    # For simplicity, assume current_age - min_age = experience
    age = period + min_age

    # Determinisctic component of income depending on experience:
    # constant + alpha_1 * age + alpha_2 * age**2
    exp_coeffs = jnp.array([constant, exp, exp_squared])
    labor_income = exp_coeffs @ (age ** jnp.arange(len(exp_coeffs)))
    working_income = jnp.exp(labor_income + wage_shock)

    return (1 - lagged_choice) * working_income


@pytest.fixture
def model_config():
    model_config = {
        "n_periods": 5,
        "choices": np.arange(2),
        "continuous_states": {
            "assets_end_of_period": np.arange(0, 100, 5, dtype=float),
        },
        "n_quad_points": 5,
    }

    return model_config


def test_sim_and_sol_model(model_config):
    params, model_specs, _ = toy_models.load_example_params_model_specs_and_config(
        "dcegm_paper_retirement_with_shocks"
    )

    model_funcs = toy_models.load_example_model_functions("dcegm_paper")

    model_with_aux = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["utility_functions_final_period"],
        budget_constraint=budget_with_aux,
    )

    model_without_aux = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["utility_functions_final_period"],
        budget_constraint=budget_without_aux,
    )

    n_agents = 1_000

    states_initial = {
        "period": jnp.zeros(n_agents, dtype=int),
        "lagged_choice": jnp.zeros(n_agents, dtype=int),
        "assets_begin_of_period": jnp.ones(n_agents, dtype=float) * 10,
    }
    n_periods = model_config["n_periods"]
    seed = 132

    df_aux = model_with_aux.solve_and_simulate(
        params=params,
        states_initial=states_initial,
        seed=seed,
    )

    df_without_aux = model_without_aux.solve_and_simulate(
        params=params,
        states_initial=states_initial,
        seed=seed,
    )
    # # First check that income is in df_aux columns
    assert "income" in df_aux.columns

    # Now drop the column and check that the rest is exactly the same
    df_aux = df_aux.drop(columns=["income"])
    assert df_aux.equals(df_without_aux)
