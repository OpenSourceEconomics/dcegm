import copy
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.pre_processing.setup_model import setup_model
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.solve import get_solve_func_for_model
from toy_models.load_example_model import load_example_models

N_PERIODS = 10
N_DISCRETE_CHOICES = 2
MAX_WEALTH = 50
WEALTH_GRID_POINTS = 100
EXPERIENCE_GRID_POINTS = 6
MAX_INIT_EXPERIENCE = 1


PARAMS = {
    "beta": 0.95,
    "delta": 0.35,
    "rho": 1.95,
    "interest_rate": 0.04,
    "lambda": 1,  # taste shock (scale) parameter
    "sigma": 1,  # shock on labor income, standard deviation
    "constant": 0.75,
    "exp": 0.04,
    "exp_squared": -0.0002,
    "consumption_floor": 0.001,
}


@pytest.fixture(scope="module")
def test_setup():

    # =================================================================================
    # Discrete experience
    # =================================================================================

    model_funcs_discr_exp = load_example_models("with_exp")

    model_params = {
        "n_choices": N_DISCRETE_CHOICES,
        "quadrature_points_stochastic": 5,
        "n_periods": N_PERIODS,
        "max_init_experience": MAX_INIT_EXPERIENCE,
    }

    state_space_options = {
        "n_periods": N_PERIODS,
        "choices": np.arange(
            N_DISCRETE_CHOICES,
        ),
        "endogenous_states": {
            "experience": np.arange(N_PERIODS + MAX_INIT_EXPERIENCE),
        },
        "continuous_states": {
            "wealth": jnp.linspace(
                0,
                MAX_WEALTH,
                WEALTH_GRID_POINTS,
            )
        },
    }
    options_discrete = {
        "model_params": model_params,
        "state_space": state_space_options,
    }

    model_disc = setup_model(
        options=options_discrete,
        state_space_functions=model_funcs_discr_exp["state_space_functions"],
        utility_functions=model_funcs_discr_exp["utility_functions"],
        utility_functions_final_period=model_funcs_discr_exp[
            "final_period_utility_functions"
        ],
        budget_constraint=model_funcs_discr_exp["budget_constraint"],
    )

    solve_disc = get_solve_func_for_model(model_disc)
    value_disc, policy_disc, endog_grid_disc = solve_disc(PARAMS)

    # =================================================================================
    # Continuous experience
    # =================================================================================

    options_cont = copy.deepcopy(options_discrete)
    options_cont["state_space"]["continuous_states"]["experience"] = jnp.linspace(
        0, 1, EXPERIENCE_GRID_POINTS
    )
    options_cont["state_space"].pop("endogenous_states")

    model_funcs_cont_exp = load_example_models("with_cont_exp")

    model_cont = setup_model(
        options=options_cont,
        state_space_functions=model_funcs_cont_exp["state_space_functions"],
        utility_functions=model_funcs_cont_exp["utility_functions"],
        utility_functions_final_period=model_funcs_cont_exp[
            "final_period_utility_functions"
        ],
        budget_constraint=model_funcs_cont_exp["budget_constraint"],
    )

    solve_cont = get_solve_func_for_model(model_cont)
    value_cont, policy_cont, endog_grid_cont = solve_cont(PARAMS)

    return (
        model_disc,
        model_cont,
        value_disc,
        policy_disc,
        endog_grid_disc,
        value_cont,
        policy_cont,
        endog_grid_cont,
    )


def test_similate_discrete_versus_continuous_experience(test_setup):
    (
        model_disc,
        model_cont,
        value_disc,
        policy_disc,
        endog_grid_disc,
        value_cont,
        policy_cont,
        endog_grid_cont,
    ) = test_setup

    n_agents = 100_000

    states_initial = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "experience": np.ones(n_agents),
    }
    wealth_initial = np.ones(n_agents) * 10

    result_disc = simulate_all_periods(
        states_initial=states_initial,
        wealth_initial=wealth_initial,
        n_periods=model_disc["options"]["state_space"]["n_periods"],
        params=PARAMS,
        seed=111,
        endog_grid_solved=endog_grid_disc,
        value_solved=value_disc,
        policy_solved=policy_disc,
        model=model_disc,
    )

    df_disc = create_simulation_df(result_disc)

    result_cont = simulate_all_periods(
        states_initial=states_initial,
        wealth_initial=wealth_initial,
        n_periods=model_cont["options"]["state_space"]["n_periods"],
        params=PARAMS,
        seed=111,
        endog_grid_solved=endog_grid_cont,
        value_solved=value_cont,
        policy_solved=policy_cont,
        model=model_cont,
    )

    df_cont = create_simulation_df(result_cont)

    # Check if taste shocks are the same
    aaae(df_disc["taste_shocks_0"], df_cont["taste_shocks_0"])
    aaae(df_disc["taste_shocks_1"], df_cont["taste_shocks_1"])

    # Check if value is reasonable close
    aaae(df_disc["value_max"], df_cont["value_max"], decimal=4)

    # Check if experience is the same
    df_cont["experience_years"] = (
        df_cont.index.get_level_values("period") + MAX_INIT_EXPERIENCE
    ) * df_cont["experience"]
    aaae(df_disc["experience"], df_cont["experience_years"])

    # Check if choices are the same
    aaae(df_disc["choice"], df_cont["choice"])

    # Check if savings and consumption are reasonable close
    aaae(df_disc["savings"], df_cont["savings"], decimal=5)
    aaae(df_disc["consumption"], df_cont["consumption"], decimal=5)
