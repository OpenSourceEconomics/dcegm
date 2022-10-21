from functools import partial
from itertools import product

import numpy as np
import pytest
from dcegm.aggregate_policy_value import calc_choice_probability
from dcegm.aggregate_policy_value import calc_current_period_value
from dcegm.egm_step import get_next_period_policy
from dcegm.egm_step import get_next_period_value
from dcegm.egm_step import sum_marginal_utility_over_choice_probs
from dcegm.interpolate import interpolate_policy
from dcegm.interpolate import interpolate_value
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
    sigma = params.loc[("shocks", "sigma"), "value"]
    consump_floor = params.loc[("assets", "consumption_floor"), "value"]
    r = params.loc[("assets", "interest_rate"), "value"]

    n_quad_points = options["quadrature_points_stochastic"]
    options["grid_points_wealth"] = n_grid_points

    child_state = np.array([period, labor_choice])
    savings_grid = np.linspace(0, max_wealth, n_grid_points)

    _quad_points, _ = roots_sh_legendre(n_quad_points)
    quad_points = norm.ppf(_quad_points) * sigma

    matrix_next_wealth = budget_constraint(
        child_state,
        savings_grid=savings_grid,
        income_shock=quad_points,
        params=params,
        options=options,
    )

    _income = _calc_stochastic_income(
        child_state,
        wage_shock=quad_points,
        params=params,
        options=options,
    )
    _income_mat = np.repeat(_income[:, np.newaxis], n_grid_points, 1)
    _savings_mat = np.full((n_quad_points, n_grid_points), savings_grid * (1 + r))

    expected_matrix = _income_mat + _savings_mat
    expected_matrix[expected_matrix < consump_floor] = consump_floor

    aaae(matrix_next_wealth, expected_matrix)


# ======================================================================================
# interpolate_policy & get_next_period_policy
# ======================================================================================


def get_inputs_and_expected_interpolate_policy(
    interest_rate, max_wealth, n_grid_points, n_quad_points
):
    _savings = np.linspace(0, max_wealth, n_grid_points)
    matrix_next_wealth = np.full(
        (n_quad_points, n_grid_points), _savings * (1 + interest_rate)
    )
    next_policy = np.tile(_savings[np.newaxis], (2, 1))

    expected_policy = np.linspace(0, max_wealth * (1 + interest_rate), n_grid_points)

    return matrix_next_wealth, next_policy, expected_policy


choice_set = [np.array([0, 1]), np.arange(7)]
interest_rate = [0.05, 0.123]
n_quad_points = [5, 10]

TEST_CASES = list(product(interest_rate, max_wealth, n_grid_points, n_quad_points))


@pytest.mark.parametrize(
    "interest_rate, max_wealth, n_grid_points, n_quad_points",
    TEST_CASES,
)
def test_interpolate_policy(interest_rate, max_wealth, n_grid_points, n_quad_points):
    (
        matrix_next_wealth,
        next_policy,
        expected_policy,
    ) = get_inputs_and_expected_interpolate_policy(
        interest_rate, max_wealth, n_grid_points, n_quad_points
    )

    policy_interp = interpolate_policy(matrix_next_wealth.flatten("F"), next_policy)
    aaae(policy_interp, np.repeat(expected_policy, n_quad_points))


TEST_CASES = list(
    product(choice_set, interest_rate, max_wealth, n_grid_points, n_quad_points)
)


@pytest.mark.parametrize(
    "choice_set, interest_rate, max_wealth, n_grid_points, n_quad_points",
    TEST_CASES,
)
def test_get_next_period_policy(
    choice_set, interest_rate, max_wealth, n_grid_points, n_quad_points
):
    options = {
        "quadrature_points_stochastic": n_quad_points,
        "grid_points_wealth": n_grid_points,
    }

    (
        matrix_next_wealth,
        _next_policy,
        _expected_policy,
    ) = get_inputs_and_expected_interpolate_policy(
        interest_rate, max_wealth, n_grid_points, n_quad_points
    )
    next_policy = np.repeat(_next_policy[np.newaxis, ...], len(choice_set), axis=0)

    policy_interp = get_next_period_policy(
        choice_set, matrix_next_wealth, next_policy, options
    )

    expected_policy = np.tile(
        np.repeat(_expected_policy, n_quad_points)[np.newaxis],
        (len(choice_set), 1),
    )
    aaae(policy_interp, expected_policy)


# ======================================================================================
# interpolate_value & get_next_period_value
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
    return np.repeat(_value, 5)


def test_interpolate_value(
    inputs_interpolate_value, value_interp_expected, load_example_model
):
    params, _ = load_example_model("retirement_no_taste_shocks")

    compute_utility = partial(
        utility_func_crra,
        params=params,
    )
    compute_value_constrained = partial(
        calc_current_period_value,
        discount_factor=params.loc[("beta", "beta"), "value"],
        compute_utility=compute_utility,
    )

    matrix_next_wealth, next_value = inputs_interpolate_value

    value_interp = interpolate_value(
        flat_wealth=matrix_next_wealth.flatten("F"),
        value=next_value,
        choice=0,
        compute_value_constrained=compute_value_constrained,
    )

    aaae(value_interp, value_interp_expected)


def test_get_next_period_value(
    inputs_interpolate_value, value_interp_expected, load_example_model
):
    params, _ = load_example_model("retirement_no_taste_shocks")

    choice_set = np.array([0])

    compute_utility = partial(
        utility_func_crra,
        params=params,
    )
    compute_value_constrained = partial(
        calc_current_period_value,
        discount_factor=params.loc[("beta", "beta"), "value"],
        compute_utility=compute_utility,
    )

    matrix_next_wealth, _next_value = inputs_interpolate_value
    next_value = np.repeat(_next_value[np.newaxis, ...], len(choice_set), axis=0)

    value_interp = get_next_period_value(
        choice_set,
        matrix_next_wealth,
        next_value,
        compute_value_constrained,
    )
    aaae(value_interp, np.tile(value_interp_expected[np.newaxis], (len(choice_set), 1)))


# ======================================================================================
# sum_marginal_utilities_over_choice_probs
# ======================================================================================

child_node_choice_set = [np.array([0]), np.array([1])]
TEST_CASES = list(product(model, child_node_choice_set, n_grid_points, n_quad_points))


@pytest.mark.parametrize(
    "model, child_node_choice_set, n_grid_points, n_quad_points",
    TEST_CASES,
)
def test_sum_marginal_utility_over_choice_probs(
    model, child_node_choice_set, n_grid_points, n_quad_points, load_example_model
):
    params, _ = load_example_model(model)

    n_choices = len(child_node_choice_set)
    n_grid_flat = n_quad_points * n_grid_points

    next_policy = np.random.rand(n_grid_flat * n_choices).reshape(
        n_choices, n_grid_flat
    )
    next_value = np.random.rand(n_grid_flat * n_choices).reshape(n_choices, n_grid_flat)

    compute_marginal_utility = partial(marginal_utility_crra, params=params)
    taste_shock_scale = params.loc[("shocks", "lambda"), "value"]

    next_marg_util = sum_marginal_utility_over_choice_probs(
        child_node_choice_set,
        next_policy,
        next_value,
        taste_shock_scale=taste_shock_scale,
        compute_marginal_utility=compute_marginal_utility,
    ).reshape((n_quad_points, n_grid_points), order="F")

    _choice_index = 0
    _choice_probabilites = calc_choice_probability(next_value, taste_shock_scale)
    _expected = _choice_probabilites[_choice_index] * compute_marginal_utility(
        next_policy[_choice_index]
    )
    expected = _expected.reshape((n_quad_points, n_grid_points), order="F")

    aaae(next_marg_util, expected)
