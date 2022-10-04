from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from dcegm.egm_step import get_next_period_wealth_matrices
from numpy.testing import assert_array_almost_equal as aaae
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model import calc_stochastic_income

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


@pytest.mark.parametrize(
    "model, period, labor_choice, max_wealth, n_grid_points", TEST_CASES
)
def test_get_next_period_wealth_matrices(
    model, period, labor_choice, max_wealth, n_grid_points
):
    params, options = get_example_model(f"{model}")
    sigma = params.loc[("shocks", "sigma"), "value"]
    n_quad_points = options["quadrature_points_stochastic"]
    options["grid_points_wealth"] = n_grid_points
    r = 0.05
    consump_floor = params.loc[("assets", "consumption_floor"), "value"]

    child_state = np.array([period, labor_choice])
    savings_grid = np.linspace(0, max_wealth, n_grid_points)

    _quad_points, _ = roots_sh_legendre(n_quad_points)
    quad_points = norm.ppf(_quad_points) * sigma

    compute_income = partial(
        calc_stochastic_income, wage_shock=quad_points, params=params, options=options
    )

    matrix_next_period_wealth = get_next_period_wealth_matrices(
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

    aaae(matrix_next_period_wealth, expected_matrix)
