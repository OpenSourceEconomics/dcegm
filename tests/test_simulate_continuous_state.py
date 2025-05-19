import copy

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm.toy_models as toy_models
from dcegm.backward_induction import get_solve_func_for_model
from dcegm.pre_processing.setup_model import create_model_dict
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods


@pytest.fixture(scope="module")
def test_setup():

    # =================================================================================
    # Discrete experience
    # =================================================================================

    model_funcs_discrete = toy_models.load_example_model_functions("with_exp")
    # params are actually the same for both models. Just name them params.
    params, options_discrete = toy_models.load_example_params_model_specs_and_config(
        "with_exp"
    )

    model_disc = create_model_dict(
        options=options_discrete,
        state_space_functions=model_funcs_discrete["state_space_functions"],
        utility_functions=model_funcs_discrete["utility_functions"],
        utility_functions_final_period=model_funcs_discrete[
            "utility_functions_final_period"
        ],
        budget_constraint=model_funcs_discrete["budget_constraint"],
    )

    solve_disc = get_solve_func_for_model(model_disc)
    value_disc, policy_disc, endog_grid_disc = solve_disc(params)

    # =================================================================================
    # Continuous experience
    # =================================================================================

    model_funcs_cont_exp = toy_models.load_example_model_functions("with_cont_exp")
    _, options_cont = toy_models.load_example_params_model_specs_and_config(
        "with_cont_exp"
    )

    model_cont = create_model_dict(
        options=options_cont,
        state_space_functions=model_funcs_cont_exp["state_space_functions"],
        utility_functions=model_funcs_cont_exp["utility_functions"],
        utility_functions_final_period=model_funcs_cont_exp[
            "utility_functions_final_period"
        ],
        budget_constraint=model_funcs_cont_exp["budget_constraint"],
    )

    solve_cont = get_solve_func_for_model(model_cont)
    value_cont, policy_cont, endog_grid_cont = solve_cont(params)

    return (
        options_discrete,
        params,
        model_disc,
        model_cont,
        value_disc,
        policy_disc,
        endog_grid_disc,
        value_cont,
        policy_cont,
        endog_grid_cont,
    )


def test_simulate_discrete_versus_continuous_experience(test_setup):
    (
        options_discrete,
        params,
        model_disc,
        model_cont,
        value_disc,
        policy_disc,
        endog_grid_disc,
        value_cont,
        policy_cont,
        endog_grid_cont,
    ) = test_setup

    max_init_exp = options_discrete["model_params"]["max_init_experience"]
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
        params=params,
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
        params=params,
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
        df_cont.index.get_level_values("period") + max_init_exp
    ) * df_cont["experience"]
    aaae(df_disc["experience"], df_cont["experience_years"])

    # Check if choices are the same
    aaae(df_disc["choice"], df_cont["choice"])

    # Check if savings and consumption are reasonable close
    aaae(df_disc["savings"], df_cont["savings"], decimal=5)
    aaae(df_disc["consumption"], df_cont["consumption"], decimal=5)
