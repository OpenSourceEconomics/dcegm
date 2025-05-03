import jax.numpy as jnp
import numpy as np
import pytest

from dcegm.pre_processing.setup_model import setup_model
from dcegm.sim_interface import get_sol_and_sim_func_for_model
from dcegm.simulation.sim_utils import create_simulation_df
from toy_models.load_example_model import load_example_models


def budget_with_aux(
    period,
    lagged_choice,
    savings_end_of_previous_period,
    income_shock_previous_period,
    options,
    params,
):
    wealth, _shock, income = budget_constraint_raw(
        period,
        lagged_choice,
        savings_end_of_previous_period,
        income_shock_previous_period,
        options,
        params,
    )
    aux_dict = {
        "income": income,
    }
    return wealth, aux_dict


def budget_without_aux(
    period,
    lagged_choice,
    savings_end_of_previous_period,
    income_shock_previous_period,
    options,
    params,
):
    wealth, _, _ = budget_constraint_raw(
        period,
        lagged_choice,
        savings_end_of_previous_period,
        income_shock_previous_period,
        options,
        params,
    )
    return wealth


def budget_constraint_raw(
    period,
    lagged_choice,
    savings_end_of_previous_period,
    income_shock_previous_period,
    options,
    params,
):
    # Calculate stochastic labor income
    income_from_previous_period = _calc_stochastic_income(
        period=period,
        lagged_choice=lagged_choice,
        wage_shock=income_shock_previous_period,
        min_age=options["min_age"],
        constant=params["constant"],
        exp=params["exp"],
        exp_squared=params["exp_squared"],
    )

    wealth_beginning_of_period = (
        income_from_previous_period
        + (1 + params["interest_rate"]) * savings_end_of_previous_period
    )

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
def state_space_options():
    state_space_options = {
        "n_periods": 5,
        "choices": np.arange(2),
        "continuous_states": {
            "wealth": np.arange(0, 100, 5, dtype=float),
        },
    }

    return state_space_options


def test_sim_and_sol_model(state_space_options, load_replication_params_and_specs):
    params, model_specs = load_replication_params_and_specs("retirement_taste_shocks")

    model_funcs = load_example_models("dcegm_paper")

    options_sol = {
        "state_space": state_space_options,
        "model_params": model_specs,
    }

    model_with_aux = setup_model(
        options=options_sol,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["final_period_utility_functions"],
        budget_constraint=budget_with_aux,
    )

    model_without_aux = setup_model(
        options=options_sol,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["final_period_utility_functions"],
        budget_constraint=budget_without_aux,
    )

    n_agents = 1_000

    states_initial = {
        "period": jnp.zeros(n_agents, dtype=int),
        "lagged_choice": jnp.zeros(n_agents, dtype=int),
    }
    n_periods = options_sol["state_space"]["n_periods"]
    seed = 132

    sim_func_aux = get_sol_and_sim_func_for_model(
        model=model_with_aux,
        states_initial=states_initial,
        wealth_initial=jnp.ones(n_agents, dtype=float) * 10,
        n_periods=n_periods,
        seed=seed,
    )
    output_dict_aux = sim_func_aux(params)
    df_aux = create_simulation_df(output_dict_aux["sim_dict"])

    sim_func_without_aux = get_sol_and_sim_func_for_model(
        model=model_without_aux,
        states_initial=states_initial,
        wealth_initial=np.ones(n_agents, dtype=float) * 10,
        n_periods=n_periods,
        seed=seed,
    )
    output_dict_without_aux = sim_func_without_aux(params)
    df_without_aux = create_simulation_df(output_dict_without_aux["sim_dict"])

    # # First check that income is in df_aux columns
    assert "income" in df_aux.columns

    # Now drop the column and check that the rest is exactly the same
    df_aux = df_aux.drop(columns=["income"])
    assert df_aux.equals(df_without_aux)
