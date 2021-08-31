import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from dcegm.consumption_retirement_model import compute_expected_value
from dcegm.consumption_retirement_model import compute_next_period_marginal_utility
from dcegm.consumption_retirement_model import inverse_marginal_utility_crra
from dcegm.consumption_retirement_model import utility_func_crra
from dcegm.solve import solve_dcegm
from numpy.testing import assert_array_almost_equal as aaae

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


def get_example_model(model):
    """Return parameters and options of an example model."""
    params = pd.read_csv(
        TEST_RESOURCES_DIR / f"{model}.csv", index_col=["category", "name"]
    )
    options = yaml.safe_load((TEST_RESOURCES_DIR / f"{model}.yaml").read_text())
    return params, options


@pytest.fixture()
def utility_functions():
    """Return dict with utility functions."""
    return {
        "utility": utility_func_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
        "next_period_marginal_utility": compute_next_period_marginal_utility,
    }


@pytest.mark.parametrize(
    "model, choice_range",
    [("deaton", [0]), ("retirement_taste_shocks", [1, 0])],
)
def test_benchmark_models(model, choice_range, utility_functions):
    params, options = get_example_model(f"{model}")

    policy_calculated, value_calculated = solve_dcegm(
        params,
        options,
        utility_functions,
        compute_expected_value,
    )

    policy_expected = pickle.load(
        open(TEST_RESOURCES_DIR / f"policy_{model}.pkl", "rb")
    )
    value_expected = pickle.load(open(TEST_RESOURCES_DIR / f"value_{model}.pkl", "rb"))

    for period in range(23, -1, -1):
        for choice in choice_range:
            if model == "deaton":
                policy_expec = policy_expected[period, choice]
                value_expec = value_expected[period, choice]
            else:
                policy_expec = policy_expected[period][choice].T
                value_expec = value_expected[period][choice].T

            aaae(
                policy_calculated[period, choice, :][
                    :,
                    ~np.isnan(policy_calculated[period, choice, :]).any(axis=0),
                ],
                policy_expec,
            )
            aaae(
                value_calculated[period, choice, :][
                    :,
                    ~np.isnan(value_calculated[period, choice, :]).any(axis=0),
                ],
                value_expec,
            )
