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

from dcegm.egm.interpolate_marginal_utility import (
    interp2d_value_and_marg_util_for_state_choice,
    interpolate_value_and_marg_util,
)
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from tests.utils.interp1d_auxiliary import (
    linear_interpolation_with_extrapolation,
    linear_interpolation_with_inserting_missing_values,
)
from tests.utils.interp2d_auxiliary import (
    custom_interp2d_quad,
    custom_interp2d_quad_value_function,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
    utility_crra,
)

PARAMS = {"beta": 0.95, "rho": 0.5, "delta": -1}


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


def test_interp2d():
    """Test from sebecker.

    This test ensures that the 2d interpolation runs properly.

    """

    np.random.seed(1234)

    for _ in range(20):
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

        test_x = np.random.uniform(30, 40, 100)
        test_y = np.random.choice(regular_grid, 100)
        test_points = np.column_stack((test_x, test_y))

        # transform input values for scipy griddata interpolation routine
        griddata_grids = np.column_stack(
            (irregular_grids.flatten(), regular_grids.flatten())
        )
        griddata_true_values = policy.flatten()

        policy_interp_scipy = griddata(
            griddata_grids, griddata_true_values, test_points, method="linear"
        )

        policy_interp_custom = custom_interp2d_quad(
            irregular_grids, regular_grid, policy, test_points
        )
        value_interp_custom = custom_interp2d_quad_value_function(
            irregular_grids,
            regular_grid,
            values=value,
            points=test_points,
            flow_util=utility_crra,
            params=PARAMS,
        )

        x_grids_jax = jnp.array(irregular_grids)
        y_grid_jax = jnp.array(regular_grid)
        policy_jax = jnp.array(policy)
        value_jax = jnp.array(value)
        test_x_jax = jnp.array(test_x)
        test_y_jax = jnp.array(test_y)

        interp2d_partial = (
            lambda x_in, y_in: interp2d_policy_and_value_on_wealth_and_regular_grid(
                regular_grid=y_grid_jax,
                wealth_grid=x_grids_jax,
                policy_grid=policy_jax,
                value_grid=value_jax,
                wealth_point_to_interp=x_in,
                regular_point_to_interp=y_in,
                compute_utility=utility_crra,
                state_choice_vec={"choice": 0},
                params=PARAMS,
            )
        )

        policy_interp_jax, value_interp_jax = jax.vmap(interp2d_partial)(
            test_x_jax,
            test_y_jax,
        )

        aaae(policy_interp_custom, policy_interp_scipy, decimal=7)
        aaae(policy_interp_custom, policy_interp_jax, decimal=7)

        aaae(policy_interp_jax, policy_interp_scipy, decimal=7)
        aaae(value_interp_jax, value_interp_custom, decimal=7)


def test_interp2d_value_and_marg_util():

    pass
