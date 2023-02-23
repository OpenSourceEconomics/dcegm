from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.interpolate import interpolate_value
from dcegm.pre_processing import calc_current_value
from dcegm.pre_processing import params_todict
from numpy.testing import assert_array_almost_equal as aaae
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model.budget_functions import (
    _calc_stochastic_income,
)
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

# ======================================================================================
# next_period_wealth_matrices
# ======================================================================================

model = ["deaton", "retirement_taste_shocks", "retirement_no_taste_shocks"]
labor_choice = [0, 1]
period = [0, 5, 7]
max_wealth = [11, 33, 50]
n_grid_points = [101, 444, 1000]

TEST_CASES = list(product(model, period, labor_choice, max_wealth, n_grid_points))


@pytest.mark.parametrize(
    "model, period, labor_choice, max_wealth, n_grid_points", TEST_CASES
)
def test_get_next_period_wealth_matrices(
    model, period, labor_choice, max_wealth, n_grid_points, load_example_model
):
    params, options = load_example_model(f"{model}")

    params_dict = params_todict(params)

    sigma = params_dict["sigma"]
    r = params_dict["interest_rate"]
    consump_floor = params_dict["consumption_floor"]

    n_quad_points = options["quadrature_points_stochastic"]
    options["grid_points_wealth"] = n_grid_points

    child_state = np.array([period, labor_choice])
    savings_grid = np.linspace(0, max_wealth, n_grid_points)

    _quad_points, _ = roots_sh_legendre(n_quad_points)
    quad_points = norm.ppf(_quad_points) * sigma

    random_saving_ind = np.random.randint(0, n_grid_points)
    random_shock_ind = np.random.randint(0, n_quad_points)

    wealth_next_period = budget_constraint(
        child_state,
        saving=savings_grid[random_saving_ind],
        income_shock=quad_points[random_shock_ind],
        params_dict=params_dict,
        options=options,
    )

    _income = _calc_stochastic_income(
        child_state,
        wage_shock=quad_points[random_shock_ind],
        params_dict=params_dict,
        options=options,
    )

    expected_budget = (1 + r) * savings_grid[random_saving_ind] + _income

    aaae(wealth_next_period, max(consump_floor, expected_budget))


# ======================================================================================
# interpolate_policy
# ======================================================================================

#
# def get_inputs_and_expected_interpolate_policy(
#     interest_rate, max_wealth, n_grid_points, n_quad_points
# ):
#     _savings = np.linspace(0, max_wealth, n_grid_points)
#     matrix_next_wealth = np.full(
#         (n_quad_points, n_grid_points), _savings * (1 + interest_rate)
#     )
#     next_policy = np.tile(_savings[np.newaxis], (2, 1))
#
#     expected_policy = np.linspace(0, max_wealth * (1 + interest_rate), n_grid_points)
#
#     return matrix_next_wealth, next_policy, expected_policy
#
#
# choice_set = [np.array([0, 1]), np.arange(7)]
# interest_rate = [0.05, 0.123]
# n_quad_points = [5, 10]
#
# TEST_CASES = list(product(interest_rate, max_wealth, n_grid_points, n_quad_points))
#
#
# @pytest.mark.parametrize(
#     "interest_rate, max_wealth, n_grid_points, n_quad_points",
#     TEST_CASES,
# )
# def test_interpolate_policy(interest_rate, max_wealth, n_grid_points, n_quad_points):
#     (
#         matrix_next_wealth,
#         next_policy,
#         expected_policy,
#     ) = get_inputs_and_expected_interpolate_policy(
#         interest_rate, max_wealth, n_grid_points, n_quad_points
#     )
#
#     random_saving_ind = np.random.randint(0, n_grid_points)
#
#     policy_interp = interpolate_policy(
#         matrix_next_wealth[0, random_saving_ind], next_policy
#     )
#
#     aaae(policy_interp, expected_policy[random_saving_ind], decimal=4)
#
#
# TEST_CASES = list(
#     product(choice_set, interest_rate, max_wealth, n_grid_points, n_quad_points)
# )


# ======================================================================================
# interpolate_value
# ======================================================================================


@pytest.fixture()
def inputs_interpolate_value():
    max_wealth = 50
    n_grid_points = 10
    n_quad_points = 5
    interest_rate = 0.05

    _savings = np.linspace(1, max_wealth, n_grid_points)
    matrix_next_wealth = np.full(
        (n_quad_points, n_grid_points), _savings * (1 + interest_rate)
    )
    next_value = np.stack([_savings, np.linspace(0, 10, n_grid_points)])

    return matrix_next_wealth, next_value


@pytest.fixture()
def value_interp_expected():
    _value = np.array(
        [
            -0.30232329,
            1.17687075,
            2.34353741,
            3.51020408,
            4.67687075,
            5.84353741,
            7.01020408,
            8.17687075,
            9.34353741,
            10.51020408,
        ]
    )
    return _value


def test_interpolate_value(
    inputs_interpolate_value, value_interp_expected, load_example_model
):
    params, _ = load_example_model("retirement_no_taste_shocks")
    params_dict = params_todict(params)

    compute_utility = jax.jit(
        partial(
            utility_func_crra,
            params_dict=params_dict,
        )
    )
    compute_value = jax.jit(
        partial(
            calc_current_value,
            discount_factor=params_dict["beta"],
            compute_utility=compute_utility,
        )
    )

    matrix_next_wealth, next_value = inputs_interpolate_value

    random_saving_ind = np.random.randint(0, matrix_next_wealth.shape[1])

    value_interp = interpolate_value(
        wealth=matrix_next_wealth[0, random_saving_ind],
        value=next_value,
        choice=0,
        compute_value=compute_value,
    )

    aaae(value_interp, value_interp_expected[random_saving_ind])


# ======================================================================================
# sum_marginal_utilities_over_choice_probs
# ======================================================================================

# child_node_choice_set = [np.array([0]), np.array([1])]
# TEST_CASES = list(product(model, child_node_choice_set, n_grid_points, n_quad_points))


# @pytest.mark.parametrize(
#     "model, child_node_choice_set, n_grid_points, n_quad_points",
#     TEST_CASES,
# )
# def test_sum_marginal_utility_over_choice_probs(
#     model, child_node_choice_set, n_grid_points, n_quad_points, load_example_model
# ):
#     params, _ = load_example_model(model)
#     params_dict = params_todict(params)
#
#     n_choices = len(child_node_choice_set)
#     n_grid_flat = n_quad_points * n_grid_points
#
#     next_policy = np.random.rand(n_grid_flat * n_choices).reshape(
#         n_choices, n_grid_flat
#     )
#     next_value = np.random.rand(n_grid_flat * n_choices).reshape(n_choices, n_grid_flat)
#
#     compute_marginal_utility = jax.jit(
#         partial(marginal_utility_crra, params_dict=params_dict)
#     )
#     taste_shock_scale = params_dict["lambda"]
#
#     next_marg_util, _ = get_child_state_marginal_util(
#         choice_set_indices=jnp.ones(n_choices, dtype=jnp.int32),
#         next_period_policy=next_policy,
#         next_period_value=next_value,
#         taste_shock_scale=taste_shock_scale,
#         compute_marginal_utility=compute_marginal_utility,
#     )
#
#     next_marg_util = next_marg_util.reshape((n_quad_points, n_grid_points), order="F")
#
#     _choice_index = 0
#     _choice_probabilites = calc_choice_probability(
#         next_value, jnp.ones(n_choices, dtype=jnp.int32), taste_shock_scale
#     )
#     _expected = _choice_probabilites[_choice_index] * compute_marginal_utility(
#         next_policy[_choice_index]
#     )
#     expected = _expected.reshape((n_quad_points, n_grid_points), order="F")
#
#     aaae(next_marg_util, expected, decimal=2)
