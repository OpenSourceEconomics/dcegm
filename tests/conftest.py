import glob
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from jax import config

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"

# Add the utils directory to the path so that we can import helper functions.
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


@pytest.fixture()
def load_example_model():
    def load_options_and_params(model):
        """Return parameters and options of an example model."""
        params = pd.read_csv(
            TEST_RESOURCES_DIR / f"{model}.csv", index_col=["category", "name"]
        )
        options = yaml.safe_load((TEST_RESOURCES_DIR / f"{model}.yaml").read_text())
        return params, options

    return load_options_and_params


def pytest_sessionstart(session):  # noqa: ARG001
    config.update("jax_enable_x64", val=True)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    # Get the current working directory
    cwd = os.getcwd()

    # Search for .npy files that match the naming pattern
    pattern = os.path.join(cwd, "[endog_grid_, policy_, value_]*.npy")
    npy_files = glob.glob(pattern)

    # Delete the matching .npy files
    for file in npy_files:
        os.remove(file)
