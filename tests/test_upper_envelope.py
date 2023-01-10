from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dcegm.pre_processing import calc_current_value
from dcegm.upper_envelope import upper_envelope
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


@pytest.fixture()
def test_data():
    choice = 0
    n_grid_wealth = 500

    _index = pd.MultiIndex.from_tuples(
        [("utility_function", "theta"), ("delta", "delta")],
        names=["category", "name"],
    )
    params = pd.DataFrame(data=[1.95, 0.35], columns=["value"], index=_index)
    discount_factor = 0.95

    compute_utility = partial(utility_func_crra, params=params)
    compute_value = partial(
        calc_current_value,
        discount_factor=discount_factor,
        compute_utility=compute_utility,
    )

    policy = np.genfromtxt(TEST_RESOURCES_DIR / "policy_egm_i.csv", delimiter=",")
    value = np.genfromtxt(TEST_RESOURCES_DIR / "value_egm_i.csv", delimiter=",")

    expected_policy = np.genfromtxt(
        TEST_RESOURCES_DIR / "policy_upper_envelope_i.csv", delimiter=","
    )
    expected_value = np.genfromtxt(
        TEST_RESOURCES_DIR / "value_upper_envelope_i.csv", delimiter=","
    )

    return (
        policy,
        value,
        choice,
        n_grid_wealth,
        compute_value,
        expected_policy,
        expected_value,
    )


def test_upper_envelope(test_data):
    (
        policy,
        value,
        choice,
        n_grid_wealth,
        compute_value,
        expected_policy,
        expected_value,
    ) = test_data

    got_policy, got_value = upper_envelope(
        policy=policy,
        value=value,
        choice=choice,
        n_grid_wealth=n_grid_wealth,
        compute_value=compute_value,
    )

    aaae(got_policy, expected_policy)
    aaae(got_value, expected_value)
