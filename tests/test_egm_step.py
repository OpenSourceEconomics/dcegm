from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from dcegm.egm_step import get_next_period_policy
from dcegm.egm_step import get_next_period_value
from dcegm.egm_step import interpolate_policy
from dcegm.egm_step import interpolate_value
from dcegm.egm_step import sum_marginal_utility_over_choice_probs
from numpy.testing import assert_array_almost_equal as aaae
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model import calc_next_period_choice_probs
from toy_models.consumption_retirement_model import calc_next_period_wealth_matrices
from toy_models.consumption_retirement_model import calc_stochastic_income
from toy_models.consumption_retirement_model import calc_value_constrained
from toy_models.consumption_retirement_model import marginal_utility_crra
from toy_models.consumption_retirement_model import utility_func_crra

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


def get_example_model(model):
    """Return parameters and options of an example model."""
    params = pd.read_csv(
        TEST_RESOURCES_DIR / f"{model}.csv", index_col=["category", "name"]
    )
    options = yaml.safe_load((TEST_RESOURCES_DIR / f"{model}.yaml").read_text())
    return params, options


model = ["deaton", "retirement_taste_shocks", "retirement_no_taste_shocks"]
labor_choice = [0, 1]
period = [0, 5, 7]
max_wealth = [11, 33, 50]
n_grid_points = [101, 444, 1000]
TEST_CASES = list(product(model, period, labor_choice, max_wealth, n_grid_points))


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


@pytest.mark.parametrize(
    "model, period, labor_choice, max_wealth, n_grid_points", TEST_CASES
)
def test_get_next_period_wealth_matrices(
    model, period, labor_choice, max_wealth, n_grid_points
):
    params, options = get_example_model(f"{model}")
    sigma = params.loc[("shocks", "sigma"), "value"]
    consump_floor = params.loc[("assets", "consumption_floor"), "value"]
    r = params.loc[("assets", "interest_rate"), "value"]

    n_quad_points = options["quadrature_points_stochastic"]
    options["grid_points_wealth"] = n_grid_points

    child_state = np.array([period, labor_choice])
    savings_grid = np.linspace(0, max_wealth, n_grid_points)

    _quad_points, _ = roots_sh_legendre(n_quad_points)
    quad_points = norm.ppf(_quad_points) * sigma

    compute_income = partial(
        calc_stochastic_income, wage_shock=quad_points, params=params, options=options
    )

    matrix_next_wealth = calc_next_period_wealth_matrices(
        child_state,
        savings_grid,
        params,
        options,
        compute_income=compute_income,
    )

    _income = compute_income(child_state)
    _income_mat = np.repeat(_income[:, np.newaxis], n_grid_points, 1)
    _savings_mat = np.full((n_quad_points, n_grid_points), savings_grid * (1 + r))

    expected_matrix = _income_mat + _savings_mat
    expected_matrix[expected_matrix < consump_floor] = consump_floor

    aaae(matrix_next_wealth, expected_matrix)


choice_set = [np.array([0]), np.array([0, 1]), np.arange(7)]
interest_rate = [0.05, 0.123]
n_quad_points = [5, 10]
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

    _savings = np.linspace(0, max_wealth, n_grid_points)
    matrix_next_wealth = np.full(
        (n_quad_points, n_grid_points), _savings * (1 + interest_rate)
    )
    next_policy = np.tile(_savings[np.newaxis], (len(choice_set), 2, 1))

    policy_interp = get_next_period_policy(
        choice_set, matrix_next_wealth, next_policy, options
    )

    _expected = np.linspace(0, max_wealth * (1 + interest_rate), n_grid_points)
    expected = np.tile(
        np.repeat(_expected, n_quad_points)[np.newaxis],
        (len(choice_set), 1),
    )
    aaae(policy_interp, expected)


TEST_CASES = list(product(interest_rate, max_wealth, n_grid_points, n_quad_points))


@pytest.mark.parametrize(
    "interest_rate, max_wealth, n_grid_points, n_quad_stochastic",
    TEST_CASES,
)
def test_interpolate_policy(
    interest_rate, max_wealth, n_grid_points, n_quad_stochastic
):
    _savings = np.linspace(0, max_wealth, n_grid_points)
    matrix_next_wealth = np.full(
        (n_quad_stochastic, n_grid_points), _savings * (1 + interest_rate)
    )
    next_policy = np.tile(_savings[np.newaxis], (2, 1))

    policy_interp = interpolate_policy(matrix_next_wealth.flatten("F"), next_policy)

    _expected = np.linspace(0, max_wealth * (1 + interest_rate), n_grid_points)
    aaae(policy_interp, np.repeat(_expected, n_quad_stochastic))


def test_get_next_period_value(value_interp_expected):
    max_wealth = 50
    n_grid_points = 10
    n_quad_stochastic = 5
    interest_rate = 0.05
    params, options = get_example_model("retirement_no_taste_shocks")

    choice_set = np.array([0])
    period = 10

    compute_utility = partial(
        utility_func_crra,
        params=params,
    )
    compute_value_constrained = partial(
        calc_value_constrained,
        beta=params.loc[("beta", "beta"), "value"],
        compute_utility=compute_utility,
    )

    _savings = np.linspace(1, max_wealth, n_grid_points)
    matrix_next_period_wealth = np.full(
        (n_quad_stochastic, n_grid_points), _savings * (1 + interest_rate)
    )
    next_period_value = np.tile(
        np.stack([_savings, np.linspace(0, 10, n_grid_points)])[np.newaxis],
        (len(choice_set), 1),
    )

    value_interp = get_next_period_value(
        choice_set,
        matrix_next_period_wealth,
        next_period_value,
        period,
        options,
        compute_utility,
        compute_value_constrained,
    )
    aaae(value_interp, np.tile(value_interp_expected[np.newaxis], (len(choice_set), 1)))

    assert True


def test_interpolate_value(value_interp_expected):
    max_wealth = 50
    n_grid_points = 10
    n_quad_stochastic = 5
    interest_rate = 0.05
    params, _ = get_example_model("retirement_no_taste_shocks")

    compute_utility = partial(
        utility_func_crra,
        params=params,
    )
    compute_value_constrained = partial(
        calc_value_constrained,
        beta=params.loc[("beta", "beta"), "value"],
        compute_utility=compute_utility,
    )

    _savings = np.linspace(1, max_wealth, n_grid_points)
    matrix_next_period_wealth = np.full(
        (n_quad_stochastic, n_grid_points), _savings * (1 + interest_rate)
    )
    next_period_value = np.stack([_savings, np.linspace(0, 10, n_grid_points)])

    value_interp = interpolate_value(
        flat_wealth=matrix_next_period_wealth.flatten("F"),
        value=next_period_value,
        choice=0,
        compute_value_constrained=compute_value_constrained,
    )

    aaae(value_interp, value_interp_expected)


child_node_choice_set = [np.array([0]), np.array([1])]
TEST_CASES = list(product(child_node_choice_set, n_grid_points, n_quad_points))


@pytest.mark.parametrize(
    "child_node_choice_set, n_grid_points, n_quad_points",
    TEST_CASES,
)
def test_sum_marginal_utility_over_choice_probs(
    child_node_choice_set, n_grid_points, n_quad_points
):
    params, _ = get_example_model("retirement_no_taste_shocks")
    options = {
        "quadrature_points_stochastic": n_quad_points,
        "grid_points_wealth": n_grid_points,
    }

    n_choices = len(child_node_choice_set)
    n_grid_flat = n_quad_points * n_grid_points

    next_policy = np.random.rand(n_grid_flat * n_choices).reshape(
        n_choices, n_grid_flat
    )
    next_value = np.random.rand(n_grid_flat * n_choices).reshape(n_choices, n_grid_flat)

    compute_marginal_utility = partial(marginal_utility_crra, params=params)
    compute_next_choice_probs = partial(
        calc_next_period_choice_probs, params=params, options=options
    )

    next_marg_util = sum_marginal_utility_over_choice_probs(
        child_node_choice_set,
        next_policy,
        next_value,
        options=options,
        compute_marginal_utility=compute_marginal_utility,
        compute_next_period_choice_probs=compute_next_choice_probs,
    )

    choice_index = 0
    choice_prob = compute_next_choice_probs(next_value, choice_index)
    expected = choice_prob * compute_marginal_utility(next_policy[choice_index])

    aaae(next_marg_util, expected.reshape((n_quad_points, n_grid_points), order="F"))
