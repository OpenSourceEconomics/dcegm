import copy

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
<<<<<<< HEAD
from dcegm.backward_induction import get_solve_func_for_model
from dcegm.pre_processing.setup_model import create_model_dict
=======
>>>>>>> 83037d3d4520f2db5a2ecf22020ce1ea3851e7b8
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods


@pytest.fixture(scope="module")
def test_setup():

    # =================================================================================
    # Discrete experience
    # =================================================================================

    model_funcs_discrete = toy_models.load_example_model_functions("with_exp")
    # params are actually the same for both models. Just name them params.
<<<<<<< HEAD
    params, options_discrete = toy_models.load_example_params_model_specs_and_config(
        "with_exp"
    )

    model_disc = create_model_dict(
        options=options_discrete,
=======
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_exp")
    )

    model_disc = dcegm.setup_model(
        model_specs=model_specs,
        model_config=model_config,
>>>>>>> 83037d3d4520f2db5a2ecf22020ce1ea3851e7b8
        state_space_functions=model_funcs_discrete["state_space_functions"],
        utility_functions=model_funcs_discrete["utility_functions"],
        utility_functions_final_period=model_funcs_discrete[
            "utility_functions_final_period"
        ],
        budget_constraint=model_funcs_discrete["budget_constraint"],
    )

    model_solved_disc = model_disc.solve(params)

    # =================================================================================
    # Continuous experience
    # =================================================================================

    model_funcs_cont_exp = toy_models.load_example_model_functions("with_cont_exp")
<<<<<<< HEAD
    _, options_cont = toy_models.load_example_params_model_specs_and_config(
        "with_cont_exp"
    )

    model_cont = create_model_dict(
        options=options_cont,
=======
    _, model_specs_cont, model_config_cont = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )

    max_init_exp = model_specs_cont["max_init_experience"]

    model_cont = dcegm.setup_model(
        model_config=model_config_cont,
        model_specs=model_specs_cont,
>>>>>>> 83037d3d4520f2db5a2ecf22020ce1ea3851e7b8
        state_space_functions=model_funcs_cont_exp["state_space_functions"],
        utility_functions=model_funcs_cont_exp["utility_functions"],
        utility_functions_final_period=model_funcs_cont_exp[
            "utility_functions_final_period"
        ],
        budget_constraint=model_funcs_cont_exp["budget_constraint"],
    )

    model_solved_cont = model_cont.solve(params)

    return {
        "model_solved_disc": model_solved_disc,
        "model_solved_cont": model_solved_cont,
        "max_init_exp": max_init_exp,
    }


def test_simulate_discrete_versus_continuous_experience(test_setup):

    n_agents = 100_000

    states_initial = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "experience": np.ones(n_agents),
        "assets_begin_of_period": np.ones(n_agents) * 10,
    }

    df_disc = test_setup["model_solved_disc"].simulate(
        states_initial=states_initial,
        seed=111,
    )

    df_cont = test_setup["model_solved_cont"].simulate(
        states_initial=states_initial,
        seed=111,
    )

    # Check if taste shocks are the same
    aaae(df_disc["taste_shocks_0"], df_cont["taste_shocks_0"])
    aaae(df_disc["taste_shocks_1"], df_cont["taste_shocks_1"])

    # Check if value is reasonable close
    aaae(df_disc["value_max"], df_cont["value_max"], decimal=4)

    # Check if experience is the same
    df_cont["experience_years"] = (
        df_cont.index.get_level_values("period") + test_setup["max_init_exp"]
    ) * df_cont["experience"]
    aaae(df_disc["experience"], df_cont["experience_years"])

    # Check if choices are the same
    aaae(df_disc["choice"], df_cont["choice"])

    # Check if savings and consumption are reasonable close
    aaae(df_disc["savings"], df_cont["savings"], decimal=5)
    aaae(df_disc["consumption"], df_cont["consumption"], decimal=5)
