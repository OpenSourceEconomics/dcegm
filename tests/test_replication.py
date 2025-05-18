import pickle
from pathlib import Path

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm.toy_models as toy_models
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import get_solve_func_for_model
from tests.utils.interp1d_auxiliary import (
    interpolate_policy_and_value_on_wealth_grid,
    linear_interpolation_with_extrapolation,
)

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"


@pytest.mark.parametrize(
    "model_name",
    [
        "retirement_no_shocks",
        "retirement_with_shocks",
        "deaton",
    ],
)
def test_benchmark_models(model_name):
    if model_name == "deaton":
        model_funcs = toy_models.load_example_model_functions("dcegm_paper_deaton")
    else:
        model_funcs = toy_models.load_example_model_functions("dcegm_paper")

    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_" + model_name
        )
    )

    model = setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_funcs,
    )

    value, policy, endog_grid = get_solve_func_for_model(model)(params)

    policy_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / f"{model_name}" / "policy.pkl").open("rb")
    )
    value_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / f"{model_name}" / "value.pkl").open("rb")
    )
    state_choice_space = model["model_structure"]["state_choice_space"]
    state_choice_space_to_test = state_choice_space[state_choice_space[:, 0] < 24]

    for state_choice_idx in range(state_choice_space_to_test.shape[0] - 1, -1, -1):
        choice = state_choice_space_to_test[state_choice_idx, -1]
        period = state_choice_space_to_test[state_choice_idx, 0]
        if model_name == "deaton":
            policy_expec = policy_expected[period, choice]
            value_expec = value_expected[period, choice]
        else:
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
