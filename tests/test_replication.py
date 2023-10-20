import pickle
from pathlib import Path

import jax.numpy as jnp
import pytest
from dcegm.interpolation import interpolate_policy_and_value_on_wealth_grid
from dcegm.interpolation import linear_interpolation_with_extrapolation
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.solve import solve_dcegm
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period_scalar,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    update_state,
)
from toy_models.consumption_retirement_model.utility_functions import (
    inverse_marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_func_log_crra,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"


@pytest.fixture()
def utility_functions():
    """Return dict with utility functions."""
    return {
        "utility": utility_func_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
        "marginal_utility": marginal_utility_crra,
    }


@pytest.fixture()
def state_space_functions():
    """Return dict with utility functions."""
    return {
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
        "update_endog_state_by_state_and_choice": update_state,
    }


@pytest.mark.parametrize(
    "model",
    [
        "retirement_no_taste_shocks",
        "retirement_taste_shocks",
        "deaton",
    ],
)
def test_benchmark_models(
    model,
    utility_functions,
    state_space_functions,
    load_example_model,
):
    options = {}
    params, _raw_options = load_example_model(f"{model}")

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

    if params.loc[("utility_function", "theta"), "value"] == 1:
        utility_functions["utility"] = utiility_func_log_crra

    (
        _model_funcs,
        _compute_upper_envelope,
        get_state_specific_choice_set,
        update_endog_state_by_state_and_choice,
    ) = process_model_functions(
        options,
        user_utility_functions=utility_functions,
        user_budget_constraint=budget_constraint,
        user_final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
    )

    (
        period_specific_state_objects,
        _state_space,
        map_state_choice_to_index,
        _,
    ) = create_state_space_and_choice_objects(
        options=options,
        get_state_specific_choice_set=get_state_specific_choice_set,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    value, policy_left, policy_right, endog_grid = solve_dcegm(
        params,
        options,
        exog_savings_grid=exog_savings_grid,
        utility_functions=utility_functions,
        budget_constraint=budget_constraint,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
    )

    policy_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / f"{model}" / "policy.pkl").open("rb")
    )
    value_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / f"{model}" / "value.pkl").open("rb")
    )

    for period in range(23, -1, -1):
        period_state_choice_dict = period_specific_state_objects[period][
            "state_choice_mat"
        ]

        for state_choice_idx_period, choice in enumerate(
            period_state_choice_dict["choice"]
        ):
            if model == "deaton":
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

            state_choice_tuple = (
                period_state_choice_dict["period"][state_choice_idx_period],
                period_state_choice_dict["lagged_choice"][state_choice_idx_period],
                period_state_choice_dict["dummy_exog"][state_choice_idx_period],
                choice,
            )
            state_choice_idx = map_state_choice_to_index[state_choice_tuple]
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
