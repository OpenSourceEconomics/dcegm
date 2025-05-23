import glob
import os
import sys
from pathlib import Path

import jax
import pytest
import yaml

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"

# Add the utils directory to the path so that we can import helper functions.
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


def pytest_sessionstart(session):  # noqa: ARG001
    jax.config.update("jax_enable_x64", val=True)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    # Get the current working directory
    cwd = os.getcwd()

    # Search for .npy files that match the naming pattern
    pattern = os.path.join(cwd, "[endog_grid_, policy_, value_]*.npy")
    npy_files = glob.glob(pattern)

    # Delete the matching .npy files
    for file in npy_files:
        os.remove(file)
