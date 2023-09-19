from pathlib import Path

import numpy as np
import pytest
from dcegm.fast_upper_envelope import fast_upper_envelope
from dcegm.fast_upper_envelope import fast_upper_envelope_wrapper
from dcegm.interpolation import interpolate_policy_and_value_on_wealth_grid
from dcegm.interpolation import linear_interpolation_with_extrapolation
from dcegm.process_model import _get_function_with_filtered_args_and_kwargs
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra
from utils.fast_upper_envelope_org import fast_upper_envelope_wrapper_org
from utils.upper_envelope_fedor import upper_envelope

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


@pytest.fixture
def setup_model():
    max_wealth = 50
    n_grid_wealth = 500
    exog_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    params = {}
    params["beta"] = 0.95  # discount_factor
    params["theta"] = 1.95
    params["delta"] = 0.35

    _state_vars = ["period", "lagged_choice", "exog_state"]
    _state_vars_to_index = {key: idx for idx, key in enumerate(_state_vars)}

    state_choice_vec = [23, 0, 0, 0]  # has to be a list!

    compute_utility = _get_function_with_filtered_args_and_kwargs(
        utility_func_crra, state_vars_to_index=_state_vars_to_index
    )

    return params, exog_savings_grid, state_choice_vec, compute_utility


@pytest.mark.parametrize("period", [2, 4, 9, 10, 18])
def test_fast_upper_envelope_wrapper(period, setup_model):
    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"period_tests/val{period}.csv", delimiter=","
    )
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"period_tests/pol{period}.csv", delimiter=","
    )
    value_refined_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"period_tests/expec_val{period}.csv", delimiter=","
    )
    policy_refined_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"period_tests/expec_pol{period}.csv", delimiter=","
    )
    policy_expected = policy_refined_fedor[
        :, ~np.isnan(policy_refined_fedor).any(axis=0)
    ]
    value_expected = value_refined_fedor[
        :,
        ~np.isnan(value_refined_fedor).any(axis=0),
    ]

    params, _exog_savings_grid, state_choice_vec, compute_utility = setup_model

    (
        endog_grid_refined,
        policy_left_refined,
        policy_right_refined,
        value_refined,
    ) = fast_upper_envelope_wrapper(
        endog_grid=policy_egm[0, 1:],
        policy=policy_egm[1, 1:],
        value=value_egm[1, 1:],
        expected_value_zero_savings=value_egm[1, 0],
        state_choice_vec=state_choice_vec,
        params=params,
        compute_utility=compute_utility,
    )

    wealth_max_to_test = np.max(endog_grid_refined[~np.isnan(endog_grid_refined)]) + 100
    wealth_grid_to_test = np.linspace(endog_grid_refined[1], wealth_max_to_test, 1000)

    value_expec_interp = linear_interpolation_with_extrapolation(
        x_new=wealth_grid_to_test, x=value_expected[0], y=value_expected[1]
    )

    policy_expec_interp = linear_interpolation_with_extrapolation(
        x_new=wealth_grid_to_test, x=policy_expected[0], y=policy_expected[1]
    )

    (
        policy_calc_interp,
        value_calc_interp,
    ) = interpolate_policy_and_value_on_wealth_grid(
        wealth_beginning_of_period=wealth_grid_to_test,
        endog_wealth_grid=endog_grid_refined,
        policy_left_grid=policy_left_refined,
        policy_right_grid=policy_right_refined,
        value_grid=value_refined,
    )

    aaae(value_calc_interp, value_expec_interp)
    aaae(policy_calc_interp, policy_expec_interp)


def test_fast_upper_envelope_against_org_fues(setup_model):
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / "period_tests/pol10.csv", delimiter=","
    )
    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / "period_tests/val10.csv", delimiter=","
    )
    _params, exog_savings_grid, state_choice_vec, compute_utility = setup_model

    (
        endog_grid_refined,
        value_refined,
        policy_left_refined,
        policy_right_refined,
    ) = fast_upper_envelope(
        endog_grid=policy_egm[0, 1:],
        value=value_egm[1, 1:],
        policy=policy_egm[1, 1:],
        expected_value_zero_savings=value_egm[1, 0],
        num_iter=int(1.2 * value_egm.shape[1]),
    )

    endog_grid_org, policy_org, value_org = fast_upper_envelope_wrapper_org(
        endog_grid=policy_egm[0],
        policy=policy_egm[1],
        value=value_egm[1],
        exog_grid=exog_savings_grid,
        choice=state_choice_vec[-1],
        compute_utility=compute_utility,
    )

    endog_grid_expected = endog_grid_org[~np.isnan(endog_grid_org)]
    policy_expected = policy_org[~np.isnan(policy_org)]
    value_expected = value_org[~np.isnan(value_org)]

    assert np.all(np.in1d(endog_grid_expected, endog_grid_refined))
    assert np.all(np.in1d(policy_expected, policy_right_refined))
    assert np.all(np.in1d(value_expected, value_refined))


@pytest.mark.parametrize("period", [2, 4, 10, 9, 18])
def test_fast_upper_envelope_against_fedor(period, setup_model):
    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"period_tests/val{period}.csv", delimiter=","
    )
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"period_tests/pol{period}.csv", delimiter=","
    )

    params, exog_savings_grid, state_choice_vec, compute_utility = setup_model

    _policy_fedor, _value_fedor = upper_envelope(
        policy=policy_egm,
        value=value_egm,
        exog_grid=exog_savings_grid,
        state_choice_vec=state_choice_vec,
        params=params,
        compute_utility=compute_utility,
    )
    policy_expected = _policy_fedor[:, ~np.isnan(_policy_fedor).any(axis=0)]
    value_expected = _value_fedor[
        :,
        ~np.isnan(_value_fedor).any(axis=0),
    ]

    (
        endog_grid_calc,
        policy_calc_left,
        policy_calc_right,
        value_calc,
    ) = fast_upper_envelope_wrapper(
        endog_grid=policy_egm[0, 1:],
        policy=policy_egm[1, 1:],
        value=value_egm[1, 1:],
        expected_value_zero_savings=value_egm[1, 0],
        state_choice_vec=state_choice_vec,
        params=params,
        compute_utility=compute_utility,
    )
    wealth_max_to_test = np.max(endog_grid_calc[~np.isnan(endog_grid_calc)]) + 100
    wealth_grid_to_test = np.linspace(endog_grid_calc[1], wealth_max_to_test, 1000)

    value_expec_interp = linear_interpolation_with_extrapolation(
        x_new=wealth_grid_to_test, x=value_expected[0], y=value_expected[1]
    )

    policy_expec_interp = linear_interpolation_with_extrapolation(
        x_new=wealth_grid_to_test, x=policy_expected[0], y=policy_expected[1]
    )

    (
        policy_calc_interp,
        value_calc_interp,
    ) = interpolate_policy_and_value_on_wealth_grid(
        wealth_beginning_of_period=wealth_grid_to_test,
        endog_wealth_grid=endog_grid_calc,
        policy_left_grid=policy_calc_left,
        policy_right_grid=policy_calc_right,
        value_grid=value_calc,
    )
    aaae(value_calc_interp, value_expec_interp)
    aaae(policy_calc_interp, policy_expec_interp)
