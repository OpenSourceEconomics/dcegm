"""This module tests if the function params_todict from dcegm.pre_processing raises the
correct ValueError when the discount factor, taste shock scale or interest rate are not
provided in the params dictionary."""
from pathlib import Path

import pytest
from dcegm.pre_processing import params_todict

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


@pytest.mark.parametrize(
    "model",
    [
        ("retirement_no_taste_shocks"),
        ("retirement_taste_shocks"),
        ("deaton"),
    ],
)
def test_missing_interest_rate(
    model,
    load_example_model,
):
    params, options = load_example_model(f"{model}")
    params_without_interest_rate = params.drop(index=("assets", "interest_rate"))
    with pytest.raises(ValueError, match="Interest rate must be provided in params."):
        params_todict(params_without_interest_rate)


@pytest.mark.parametrize(
    "model",
    [
        ("retirement_no_taste_shocks"),
        ("retirement_taste_shocks"),
        ("deaton"),
    ],
)
def test_missing_discount_factor(
    model,
    load_example_model,
):
    params, options = load_example_model(f"{model}")
    params_without_beta = params.drop(index=("beta", "beta"))
    with pytest.raises(ValueError, match="Discount factor must be provided in params."):
        params_todict(params_without_beta)


@pytest.mark.parametrize(
    "model",
    [
        ("retirement_no_taste_shocks"),
        ("retirement_taste_shocks"),
        ("deaton"),
    ],
)
def test_missing_taste_shock_scale(
    model,
    load_example_model,
):
    params, options = load_example_model(f"{model}")
    params_without_lambda = params.drop(index=("shocks", "lambda"))
    with pytest.raises(
        ValueError, match="Taste shock scale must be provided in params."
    ):
        params_todict(params_without_lambda)
