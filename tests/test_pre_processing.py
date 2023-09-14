from pathlib import Path

import numpy as np
import pytest
from dcegm.pre_processing import convert_params_to_dict
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_func_log_crra,
)


# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


@pytest.fixture()
def test_data(load_example_model):
    n_grid = 10

    next_period_value = np.arange(n_grid)
    consumption = np.arange(n_grid) + 1

    params, _ = load_example_model("retirement_no_taste_shocks")
    params.loc[("utility_function", "theta"), "value"] = 1

    delta = params.loc[("delta", "delta"), "value"]
    beta = params.loc[("beta", "beta"), "value"]
    params = {"beta": beta, "delta": delta}

    compute_utility = utiility_func_log_crra

    return consumption, next_period_value, params, compute_utility


@pytest.mark.parametrize(
    "model",
    [
        ("retirement_no_taste_shocks"),
        ("retirement_taste_shocks"),
        ("deaton"),
    ],
)
def test_missing_beta(
    model,
    load_example_model,
):
    params, _ = load_example_model(f"{model}")
    params_without_beta = params.drop(index=("beta", "beta"))
    with pytest.raises(ValueError, match="Beta must be provided in params."):
        convert_params_to_dict(params_without_beta)
