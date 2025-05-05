import pickle
from pathlib import Path

import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import get_solve_func_for_model
from dcegm.toy_models.example_model_functions import load_example_models
from tests.utils.interp1d_auxiliary import (
    interpolate_policy_and_value_on_wealth_grid,
    linear_interpolation_with_extrapolation,
)

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"


def test_benchmark_models(load_replication_params_and_specs):
    params, model_specs = load_replication_params_and_specs("retirement_taste_shocks")
    options = {}

    options["model_params"] = model_specs
    options["model_params"]["n_choices"] = model_specs["n_discrete_choices"]
    options["state_space"] = {
        "n_periods": 25,
        "choices": [i for i in range(model_specs["n_discrete_choices"])],
        "continuous_states": {
            "wealth": jnp.linspace(
                0,
                options["model_params"]["max_wealth"],
                options["model_params"]["n_grid_points"],
            )
        },
    }

    model_funcs = load_example_models("dcegm_paper")

    shock_functions = {"taste_shock_scale_per_state": taste_shock_per_lagged_choice}

    model = setup_model(
        options=options,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["final_period_utility_functions"],
        budget_constraint=model_funcs["budget_constraint"],
        shock_functions=shock_functions,
    )

    value, policy, endog_grid = get_solve_func_for_model(model)(params)

    policy_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / "retirement_taste_shocks/policy.pkl").open(
            "rb"
        )
    )
    value_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / "retirement_taste_shocks/value.pkl").open(
            "rb"
        )
    )
    state_choice_space = model["model_structure"]["state_choice_space"]
    state_choice_space_to_test = state_choice_space[state_choice_space[:, 0] < 24]

    for state_choice_idx in range(state_choice_space_to_test.shape[0] - 1, -1, -1):
        choice = state_choice_space_to_test[state_choice_idx, -1]
        period = state_choice_space_to_test[state_choice_idx, 0]

        policy_expec = policy_expected[period][1 - choice].T
        value_expec = value_expected[period][1 - choice].T

        wealth_grid_to_test = jnp.linspace(
            policy_expec[0][1], policy_expec[0][-1] + 10, 1000
        )

        value_expec_interp = linear_interpolation_with_extrapolation(
            x_new=wealth_grid_to_test, x=value_expec[0], y=value_expec[1]
        )
        policy_expec_interp = linear_interpolation_with_extrapolation(
            x_new=wealth_grid_to_test, x=policy_expec[0], y=policy_expec[1]
        )

        (
            policy_calc_interp,
            value_calc_interp,
        ) = interpolate_policy_and_value_on_wealth_grid(
            wealth_beginning_of_period=wealth_grid_to_test,
            endog_wealth_grid=endog_grid[state_choice_idx],
            policy=policy[state_choice_idx],
            value=value[state_choice_idx],
        )

        aaae(policy_expec_interp, policy_calc_interp)
        aaae(value_expec_interp, value_calc_interp)


def taste_shock_per_lagged_choice(lagged_choice, params):
    return lagged_choice * 0.5 + (1 - lagged_choice) * params["taste_shock_scale"]
