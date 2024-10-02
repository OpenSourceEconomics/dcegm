"""This module tests the 1d and 2d linear interpolation functions:

- linear_interpolation_with_extrapolation,
- linear_interpolation_with_inserting_missing_values
- interp2d_policy_and_value_on_wealth_and_regular_grid

The results are compared to the ones from scipy's linear interpolation
function interp1d and griddata.

"""

from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal as aaae
from scipy.interpolate import griddata, interp1d

from dcegm.egm.interpolate_marginal_utility import (
    interp2d_value_and_marg_util_for_state_choice,
)
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options
from tests.utils.interp1d_auxiliary import (
    linear_interpolation_with_extrapolation,
    linear_interpolation_with_inserting_missing_values,
)
from tests.utils.interp2d_auxiliary import (
    custom_interp2d_quad,
    custom_interp2d_quad_value_function,
)
from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    marginal_utility_crra,
    utility_crra,
)


def utility_crra_with_second_continuous(
    consumption: jnp.array,
    choice: int,
    params: Dict[str, float],
) -> jnp.array:
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (jnp.array): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """

    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])

    utility = utility_consumption - (1 - choice) * params["delta"]

    return utility


PARAMS = {
    "beta": 0.95,
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

        test_x = np.random.uniform(30, 40, 44)
        test_y = np.random.choice(regular_grid, 44)
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

        compute_utility = determine_function_arguments_and_partial_options(
            utility_crra_with_second_continuous,
            options={},
            continuous_state_name="continuous_state",
        )

        value_interp_custom = custom_interp2d_quad_value_function(
            irregular_grids,
            regular_grid,
            values=value,
            points=test_points,
            flow_util=compute_utility,
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
                compute_utility=compute_utility,
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


# =====================================================================================


def test_interp2d_value_and_marg_util():

    marginal_utility_crra_partial = determine_function_arguments_and_partial_options(
        func=marginal_utility_crra,
        options={},
    )

    np.random.seed(1234)
    a, b = np.random.uniform(1, 10), np.random.uniform(1, 10)

    def functional_form(x, y):
        return a + np.log((x + y) * b)

    wealth_grid = np.empty((6, 100))  # 6 experience levels, 100 wealth levels
    for k in range(6):
        wealth_grid[k, :] = np.sort(np.exp(np.random.uniform(1, np.log(100), 100)))

    experience_grid = np.linspace(0, 1, 6)

    _experience_grid_aux = np.column_stack([experience_grid for i in range(100)])
    policy = functional_form(wealth_grid, _experience_grid_aux)
    value = functional_form(wealth_grid, _experience_grid_aux) * np.log(2)

    wealth_next = np.random.uniform(30, 40, 100)
    experience_next = np.random.choice(experience_grid, 6)

    wealth_next_state_choice = np.tile(
        np.expand_dims(np.tile(wealth_next, (2, 6, 1)), axis=-1), (1, 1, 1, 5)
    )
    experience_next_state_choice = np.tile(experience_next, (2, 1))

    policy_state_choice = np.tile(policy, (2, 1, 1))
    value_state_choice = np.tile(value, (2, 1, 1))
    wealth_grid_state_choice = np.tile(wealth_grid, (2, 1, 1))

    interp_for_single_state_choice = vmap(
        interp2d_value_and_marg_util_for_state_choice,
        in_axes=(None, None, 0, None, 0, 0, 0, 0, 0, None),  # discrete state-choice
    )

    marg_util, val = interp_for_single_state_choice(
        marginal_utility_crra_partial,
        utility_crra,
        {"choice": jnp.array([0, 1])},
        jnp.array(experience_grid),
        jnp.array(wealth_next_state_choice),
        jnp.array(experience_next_state_choice),
        jnp.array(wealth_grid_state_choice),
        jnp.array(policy_state_choice),
        jnp.array(value_state_choice),
        PARAMS,
    )

    np.testing.assert_equal(marg_util.shape, (2, 6, 100, 5))
    np.testing.assert_equal(val.shape, (2, 6, 100, 5))
