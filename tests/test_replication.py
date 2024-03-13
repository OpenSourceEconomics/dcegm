import pickle
from pathlib import Path

import jax.numpy as jnp
import pytest
from dcegm.interpolation import interpolate_policy_and_value_on_wealth_grid
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import solve_dcegm
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.utility_functions import (
    inverse_marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_log_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_log_crra_final_consume_all,
)
from toy_models.consumption_retirement_model.utility_functions import utility_crra

from tests.utils.interpolations import linear_interpolation_with_extrapolation

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"


@pytest.fixture(scope="module")
def utility_functions():
    """Return dict with utility functions."""
    return {
        "utility": utility_crra,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
    }


@pytest.mark.parametrize(
    "model_name",
    [
        "retirement_no_taste_shocks",
        "retirement_taste_shocks",
        "deaton",
    ],
)
def test_benchmark_models(
    model_name,
    load_example_model,
    state_space_functions,
    utility_functions,
    utility_functions_final_period,
):
    options = {}
    params, _raw_options = load_example_model(f"{model_name}")

    options["model_params"] = _raw_options
    options["model_params"]["n_choices"] = _raw_options["n_discrete_choices"]
    options["state_space"] = {
        "n_periods": 25,
        "choices": [i for i in range(_raw_options["n_discrete_choices"])],
    }

    exog_savings_grid = jnp.linspace(
        0,
        options["model_params"]["max_wealth"],
        options["model_params"]["n_grid_points"],
    )

    if params["rho"] == 1:
        utility_functions["utility"] = utiility_log_crra
        utility_functions_final_period["utility"] = utiility_log_crra_final_consume_all

    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    value, policy_left, policy_right, endog_grid = solve_dcegm(
        params,
        options,
        exog_savings_grid=exog_savings_grid,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    policy_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / f"{model_name}" / "policy.pkl").open("rb")
    )
    value_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / f"{model_name}" / "value.pkl").open("rb")
    )
    state_choice_space = model["state_choice_space"]
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
            policy_left_grid=policy_left[state_choice_idx],
            policy_right_grid=policy_right[state_choice_idx],
            value_grid=value[state_choice_idx],
        )

        aaae(policy_expec_interp, policy_calc_interp)
        aaae(value_expec_interp, value_calc_interp)
