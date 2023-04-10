from functools import partial

import numpy as np
import pytest
from dcegm.pre_processing import calc_current_value
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_func_log_crra,
)


@pytest.fixture()
def test_data(load_example_model):
    n_grid = 10

    next_period_value = np.arange(n_grid)
    consumption = np.arange(n_grid) + 1

    params, _ = load_example_model("retirement_no_taste_shocks")
    params.loc[("utility_function", "theta"), "value"] = 1

    delta = params.loc[("delta", "delta"), "value"]

    beta = params.loc[("beta", "beta"), "value"]
    params_dict = {"delta": delta}
    compute_utility = partial(utiility_func_log_crra, params_dict=params_dict)

    return consumption, next_period_value, delta, beta, compute_utility


@pytest.mark.parametrize("choice", [0, 1])
def test_calc_value(choice, test_data):
    consumption, next_period_value, delta, beta, compute_utility = test_data

    expected = np.log(consumption) - (1 - choice) * delta + beta * next_period_value

    got = calc_current_value(
        consumption=consumption,
        next_period_value=next_period_value,
        choice=choice,
        discount_factor=beta,
        compute_utility=compute_utility,
    )
    aaae(got, expected)
