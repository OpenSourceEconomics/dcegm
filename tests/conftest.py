from pathlib import Path

import pandas as pd
import pytest
import yaml

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


@pytest.fixture
def load_example_model():
    def load_options_and_params(model):
        """Return parameters and options of an example model."""
        params = pd.read_csv(
            TEST_RESOURCES_DIR / f"{model}.csv", index_col=["category", "name"]
        )
        options = yaml.safe_load((TEST_RESOURCES_DIR / f"{model}.yaml").read_text())
        return params, options

    return load_options_and_params
