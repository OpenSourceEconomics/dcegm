"""This module tests the function do_upper_envelope_step from dcegm.upper_envelope_step.

For fixed inputs of do_upper_envelope_step, both outputs are compared to the true values
of current_policy and current_value.

"""
# Imports
import pickle as pkl
from pathlib import Path

import pytest
from dcegm.upper_envelope_step import do_upper_envelope_step
from numpy.testing import assert_array_almost_equal as aaae

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


@pytest.fixture()
def data():
    out = {}
    # inputs do_upper_envelope_step
    current_policy_input = pkl.load(
        open(TEST_RESOURCES_DIR / "current_policy_input.pkl", "rb")
    )
    current_value_input = pkl.load(
        open(TEST_RESOURCES_DIR / "current_value_input.pkl", "rb")
    )
    choice = 0
    n_grid_wealth = 500
    compute_value_input = pkl.load(
        open(TEST_RESOURCES_DIR / "compute_value_input.pkl", "rb")
    )

    # apply function do_upper_envelope_step to given inputs
    out["current_policy"], out["current_value"] = do_upper_envelope_step(
        current_policy_input,
        current_value_input,
        choice,
        n_grid_wealth,
        compute_value_input,
    )
    return out


# compare current_policy to true value
def test_upper_envelope_step_policy(data):
    aaae(
        data["current_policy"],  # correct outputs
        pkl.load(open(TEST_RESOURCES_DIR / "current_policy_output.pkl", "rb")),
    )


# compare current_value to true value
def test_upper_envelope_step_value(data):
    aaae(
        data["current_value"],
        pkl.load(open(TEST_RESOURCES_DIR / "current_value_output.pkl", "rb")),
    )
