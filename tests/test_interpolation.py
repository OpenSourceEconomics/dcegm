"""This module tests the 1d and 2d linear interpolation functions:

- linear_interpolation_with_extrapolation,
- linear_interpolation_with_inserting_missing_values
- interp2d_policy_and_value_on_wealth_and_regular_grid

The results are compared to the ones from scipy's linear interpolation
function interp1d and griddata.

"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal as aaae
from scipy.interpolate import griddata, interp1d

from dcegm.interpolation.interp2d import (
    interp2d_policy_on_wealth_and_regular_grid,
    interp2d_value_on_wealth_and_regular_grid,
)
from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)
from dcegm.toy_models.cons_ret_model_dcegm_paper import utility_crra
from tests.utils.interp1d_auxiliary import (
    linear_interpolation_with_extrapolation,
    linear_interpolation_with_inserting_missing_values,
)
from tests.utils.interp2d_auxiliary import (
    custom_interp2d_quad,
    custom_interp2d_quad_value_function,
)

PARAMS = {
    "discount_factor": 0.95,
    "rho": 0.5,
    "delta": -1,
    "interest_rate": 0.05,
    "consumption_floor": 100,
    "pension": 1000,
    "labor_income": 2000,
}


@pytest.fixture()
def random_test_data():
    """Draw random test arrays from a uniform distribution."""
    n = 5
    m = 7

    x = np.random.rand(n)
    y = np.random.rand(n)
    x_new = np.random.rand(m) * 2
    missing_value = np.random.rand(1)

    return x, y, x_new, missing_value


def test_linear_interpolation_with_extrapolation(random_test_data):
    x, y, x_new, _ = random_test_data

    got = linear_interpolation_with_extrapolation(x, y, x_new)
    expected = interp1d(x, y, fill_value="extrapolate")(x_new)

    assert_allclose(got, expected)


def test_linear_interpolation_with_missing_values(random_test_data):
    x, y, x_new, missing_value = random_test_data

    got = linear_interpolation_with_inserting_missing_values(x, y, x_new, missing_value)
    expected = interp1d(x, y, bounds_error=False, fill_value=missing_value)(x_new)

    assert_allclose(got, expected)


N_TEST_CASES = 20


@pytest.fixture(scope="module")
def test_cases():
    test_cases = {}
    np.random.seed(1234)

    for test_id in range(N_TEST_CASES):
        test_cases[test_id] = {}

        # setup a functional form
        a, b = np.random.uniform(1, 10), np.random.uniform(1, 10)

        def functional_form(x, y):
            return a + np.log((x + y) * b)

        # create x_grids
        irregular_grids = np.empty((10, 100))
        for k in range(10):
            irregular_grids[k, :] = np.sort(
                np.exp(np.random.uniform(1, np.log(100), 100))
            )
        regular_grid = np.linspace(1e-8, 100, 10)
        regular_grids = np.column_stack([regular_grid for i in range(100)])

        policy = functional_form(irregular_grids, regular_grids)
        value = functional_form(irregular_grids, regular_grids) * 3.5

        test_x = np.random.uniform(30, 40, 44)
        test_y = np.random.choice(regular_grid, 44)
        test_points = np.column_stack((test_x, test_y))

        # transform input values for scipy griddata interpolation routine
        griddata_grids = np.column_stack(
            (irregular_grids.flatten(), regular_grids.flatten())
        )
        griddata_true_values = policy.flatten()

        compute_utility = determine_function_arguments_and_partial_model_specs(
            utility_crra,
            model_specs={},
            continuous_state_name="continuous_state",
        )

        test_cases[test_id]["test_points"] = test_points
        test_cases[test_id]["policy"] = policy
        test_cases[test_id]["value"] = value
        test_cases[test_id]["irregular_grids"] = irregular_grids
        test_cases[test_id]["regular_grid"] = regular_grid
        test_cases[test_id]["griddata_grids"] = griddata_grids
        test_cases[test_id]["griddata_true_values"] = griddata_true_values
        test_cases[test_id]["test_x"] = test_x
        test_cases[test_id]["test_y"] = test_y
        test_cases[test_id]["compute_utility"] = compute_utility

    return test_cases


@pytest.mark.parametrize("test_id", range(N_TEST_CASES))
def test_interp2d_against_scipy(test_cases, test_id):
    """Test from sebecker.

    This test ensures that the 2d interpolation runs properly.

    """
    test_case = test_cases[test_id]
    test_points = test_case["test_points"]
    policy = test_case["policy"]
    irregular_grids = test_case["irregular_grids"]
    regular_grid = test_case["regular_grid"]
    griddata_grids = test_case["griddata_grids"]
    griddata_true_values = test_case["griddata_true_values"]
    test_x = test_case["test_x"]
    test_y = test_case["test_y"]

    policy_interp_scipy = griddata(
        griddata_grids, griddata_true_values, test_points, method="linear"
    )

    interp2d_partial = lambda x_in, y_in: interp2d_policy_on_wealth_and_regular_grid(
        regular_grid=jnp.array(regular_grid),
        wealth_grid=jnp.array(irregular_grids),
        policy_grid=jnp.array(policy),
        wealth_point_to_interp=x_in,
        regular_point_to_interp=y_in,
    )

    policy_interp_jax = jax.vmap(interp2d_partial)(
        jnp.array(test_x),
        jnp.array(test_y),
    )

    aaae(policy_interp_jax, policy_interp_scipy, decimal=7)


@pytest.mark.parametrize("test_id", range(N_TEST_CASES))
def test_interp2d_against_custom(test_cases, test_id):
    """Test interpolation function against custom interpolation function.

    We test the single interfaces as the joint ones are tested throgugh the
    `solve_dcegm` function.

    """
    test_case = test_cases[test_id]
    test_points = test_case["test_points"]
    policy = test_case["policy"]
    value = test_case["value"]
    irregular_grids = test_case["irregular_grids"]
    regular_grid = test_case["regular_grid"]
    test_x = test_case["test_x"]
    test_y = test_case["test_y"]
    compute_utility = test_case["compute_utility"]

    policy_interp_custom = custom_interp2d_quad(
        irregular_grids, regular_grid, policy, test_points
    )
    value_interp_custom = custom_interp2d_quad_value_function(
        irregular_grids,
        regular_grid,
        values=value,
        points=test_points,
        flow_util=compute_utility,
        params=PARAMS,
    )

    interp2d_partial_policy = (
        lambda x_in, y_in: interp2d_policy_on_wealth_and_regular_grid(
            regular_grid=jnp.array(regular_grid),
            wealth_grid=jnp.array(irregular_grids),
            policy_grid=jnp.array(policy),
            wealth_point_to_interp=x_in,
            regular_point_to_interp=y_in,
        )
    )

    policy_interp_jax = jax.vmap(interp2d_partial_policy)(
        jnp.array(test_x),
        jnp.array(test_y),
    )

    interp2d_partial_value = (
        lambda x_in, y_in: interp2d_value_on_wealth_and_regular_grid(
            regular_grid=jnp.array(regular_grid),
            wealth_grid=jnp.array(irregular_grids),
            value_grid=jnp.array(value),
            wealth_point_to_interp=x_in,
            regular_point_to_interp=y_in,
            compute_utility=compute_utility,
            state_choice_vec={"choice": 0},
            params=PARAMS,
            discount_factor=PARAMS["discount_factor"],
        )
    )

    value_interp_jax = jax.vmap(interp2d_partial_value)(
        jnp.array(test_x),
        jnp.array(test_y),
    )

    aaae(value_interp_custom, value_interp_jax, decimal=7)
    aaae(policy_interp_custom, policy_interp_jax, decimal=7)
