import pickle
from pathlib import Path

import numpy as np
import pytest
from dcegm.solve import solve_dcegm
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.budget_functions import marginal_wealth
from toy_models.consumption_retirement_model.final_period import solve_final_period
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_choice_set,
)
from toy_models.consumption_retirement_model.utility_functions import (
    inverse_marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra


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
def budget_functions():
    """Return dict with utility functions."""
    return {
        "budget_constraint": budget_constraint,
        "marginal_budget_constraint": marginal_wealth,
    }


@pytest.fixture()
def state_space_functions():
    """Return dict with utility functions."""
    return {
        "create_state_space": create_state_space,
        "get_state_specific_choice_set": get_state_specific_choice_set,
    }


@pytest.mark.parametrize(
    "model, choice_range",
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
    budget_functions,
    state_space_functions,
    load_example_model,
):
    params, options = load_example_model(f"{model}")

    state_space, indexer = create_state_space(options)

    policy_calculated, value_calculated = solve_dcegm(
        params,
        options,
        utility_functions,
        budget_functions=budget_functions,
        solve_final_period=solve_final_period,
        state_space_functions=state_space_functions,
    )

    policy_expected = pickle.load(
        open(TEST_RESOURCES_DIR / f"policy_{model}.pkl", "rb")
    )
    value_expected = pickle.load(open(TEST_RESOURCES_DIR / f"value_{model}.pkl", "rb"))

    for period in range(23, -1, -1):
        relevant_subset_state = state_space[np.where(state_space[:, 0] == period)][0]

        state_index = indexer[relevant_subset_state[0], relevant_subset_state[1]]
        for choice in choice_range:
            if model == "deaton":
                policy_expec = policy_expected[period, choice]
                value_expec = value_expected[period, choice]
            else:
                policy_expec = policy_expected[period][1 - choice].T
                value_expec = value_expected[period][1 - choice].T

            aaae(
                policy_calculated[state_index, choice, :][
                    :,
                    ~np.isnan(policy_calculated[state_index, choice, :]).any(axis=0),
                ],
                policy_expec,
            )
            aaae(
                value_calculated[state_index, choice, :][
                    :,
                    ~np.isnan(value_calculated[state_index, choice, :]).any(axis=0),
                ],
                value_expec,
            )
