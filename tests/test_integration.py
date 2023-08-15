import pickle
from pathlib import Path

import numpy as np
import pytest
from dcegm.interpolation import (
    interpolate_policy_and_value_on_wealth_grid,
    linear_interpolation_with_extrapolation,
)
from dcegm.solve import solve_dcegm
from dcegm.state_space import create_state_choice_space
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.exogenous_processes import (
    get_transition_matrix_by_state,
)
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period_scalar,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.utility_functions import (
    inverse_marginal_utility_crra,
    marginal_utility_crra,
    utiility_func_log_crra,
    utility_func_crra,
)

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


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
        "create_state_space": create_state_space,
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
    }


@pytest.mark.parametrize(
    ("model", "choice_range"),
    [
        ("retirement_no_taste_shocks", [0, 1]),
        ("retirement_taste_shocks", [0, 1]),
        ("deaton", [0]),
    ],
)
def test_benchmark_models(
    model,
    choice_range,
    utility_functions,
    state_space_functions,
    load_example_model,
):
    params, options = load_example_model(f"{model}")
    options["n_exog_processes"] = 1

    state_space, map_state_to_index = create_state_space(options)
    state_choice_space, *_ = create_state_choice_space(
        state_space,
        map_state_to_index,
        state_space_functions["get_state_specific_choice_set"],
    )

    if params.loc[("utility_function", "theta"), "value"] == 1:
        utility_functions["utility"] = utiility_func_log_crra

    solve_dcegm(
        params,
        options,
        utility_functions,
        budget_constraint=budget_constraint,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
        transition_function=get_transition_matrix_by_state,
    )

    policy_expected = pickle.load(
        (TEST_RESOURCES_DIR / f"policy_{model}.pkl").open("rb"),
    )
    value_expected = pickle.load((TEST_RESOURCES_DIR / f"value_{model}.pkl").open("rb"))

    # need to loop over period? Isn't state_choice space enough?
    for period in range(23, -1, -1):
        idxs_state_choice_combs = np.where(state_choice_space[:, 0] == period)[0]

        endog_grid_got = np.load(f"endog_grid_{period}.npy")
        policy_got = np.load(f"policy_{period}.npy")
        value_got = np.load(f"value_{period}.npy")

        for state_choice_idx, state_choice_vec in enumerate(idxs_state_choice_combs):
            choice = state_choice_space[state_choice_vec, -1]

            if model == "deaton":
                policy_expec = policy_expected[period, choice]
                value_expec = value_expected[period, choice]
            else:
                policy_expec = policy_expected[period][1 - choice].T
                value_expec = value_expected[period][1 - choice].T

            wealth_grid_to_test = np.linspace(
                policy_expec[0][1],
                policy_expec[0][-1] + 10,
                1000,
            )

            value_expec_interp = linear_interpolation_with_extrapolation(
                x_new=wealth_grid_to_test,
                x=value_expec[0],
                y=value_expec[1],
            )
            policy_expec_interp = linear_interpolation_with_extrapolation(
                x_new=wealth_grid_to_test,
                x=policy_expec[0],
                y=policy_expec[1],
            )

            (
                policy_calc_interp,
                value_calc_interp,
            ) = interpolate_policy_and_value_on_wealth_grid(
                begin_of_period_wealth=wealth_grid_to_test,
                endog_wealth_grid=endog_grid_got[state_choice_idx],
                policy_grid=policy_got[state_choice_idx],
                value_grid=value_got[state_choice_idx],
            )

            aaae(policy_expec_interp, policy_calc_interp)
            aaae(value_expec_interp, value_calc_interp)
