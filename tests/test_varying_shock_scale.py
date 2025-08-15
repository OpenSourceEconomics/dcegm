import pickle
from pathlib import Path

import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
from dcegm.pre_processing.setup_model import create_model_dict
from tests.utils.interp1d_auxiliary import (
    linear_interpolation_with_extrapolation,
)

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"


def test_benchmark_models():
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_retirement_with_shocks"
        )
    )

    model_funcs = toy_models.load_example_model_functions("dcegm_paper")

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_funcs,
    )
    model_solved = model.solve(params)

    policy_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / "retirement_with_shocks/policy.pkl").open(
            "rb"
        )
    )
    value_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / "retirement_with_shocks/value.pkl").open("rb")
    )
    state_choice_space = model.model_structure["state_choice_space"]
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

        state = {
            "period": period,
            "lagged_choice": state_choice_space_to_test[state_choice_idx, 1],
            "assets_begin_of_period": wealth_grid_to_test,
        }
        (
            policy_calc_interp,
            value_calc_interp,
        ) = model_solved.value_and_policy_for_states_and_choices(
            states=state,
            choices=choice,
        )

        aaae(policy_expec_interp, policy_calc_interp)
        aaae(value_expec_interp, value_calc_interp)


def taste_shock_per_lagged_choice(lagged_choice, params):
    return lagged_choice * 0.5 + (1 - lagged_choice) * params["taste_shock_scale"]
