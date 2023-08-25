from itertools import product

import numpy as np
import pytest
from dcegm.pre_processing import convert_params_to_dict
from numpy.testing import assert_array_almost_equal as aaae
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model.budget_functions import (
    _calc_stochastic_income,
)
from toy_models.consumption_retirement_model.budget_functions import budget_constraint

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

    params_dict = convert_params_to_dict(params)

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
        params=params_dict,
        options=options,
    )

    _income = _calc_stochastic_income(
        child_state,
        wage_shock=quad_points[random_shock_ind],
        params=params_dict,
        options=options,
    )

    budget_expected = (1 + r) * savings_grid[random_saving_ind] + _income

    aaae(wealth_next_period, max(consump_floor, budget_expected))
