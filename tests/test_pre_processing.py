from pathlib import Path

import numpy as np
import pytest
from dcegm.pre_processing import calc_current_value
from dcegm.pre_processing import convert_params_to_dict
from numpy.testing import assert_array_almost_equal as aaae
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


@pytest.mark.parametrize("choice", [0, 1])
def test_calc_value(choice, test_data):
    consumption, next_period_value, params, compute_utility = test_data

    expected = (
        np.log(consumption)
        - (1 - choice) * params["delta"]
        + params["beta"] * next_period_value
    )

    got = calc_current_value(
        consumption=consumption,
        next_period_value=next_period_value,
        choice=choice,
        params=params,
        compute_utility=compute_utility,
    )
    aaae(got, expected)


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
    params, _ = load_example_model(f"{model}")
    params_without_interest_rate = params.drop(index=("assets", "interest_rate"))
    with pytest.raises(ValueError, match="Interest rate must be provided in params."):
        convert_params_to_dict(params_without_interest_rate)


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
    params, _ = load_example_model(f"{model}")
    params_without_beta = params.drop(index=("beta", "beta"))
    with pytest.raises(ValueError, match="Discount factor must be provided in params."):
        convert_params_to_dict(params_without_beta)


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
    params, _options = load_example_model(f"{model}")
    params_without_lambda = params.drop(index=("shocks", "lambda"))
    with pytest.raises(
        ValueError, match="Taste shock scale must be provided in params."
    ):
        convert_params_to_dict(params_without_lambda)
