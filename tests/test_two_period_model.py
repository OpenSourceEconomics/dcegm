"""This is a test for a simple two period model with exogenous processes.

We test DC-EGM against the closed form solution of the Euler equation.

"""
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm

from tests.two_period_models.euler_equation import euler_rhs_exog_ltc
from tests.two_period_models.euler_equation import euler_rhs_exog_ltc_and_job_offer
from tests.two_period_models.model import marginal_utility


WEALTH_GRID_POINTS = 100
ALL_WEALTH_GRIDS = list(range(WEALTH_GRID_POINTS))
RANDOM_TEST_SET_LTC = np.random.choice(ALL_WEALTH_GRIDS, size=10, replace=False)
PRODUCT_EXOG_LTC = list(product(RANDOM_TEST_SET_LTC, list(range(4))))
EXOG_LTC = [(euler_rhs_exog_ltc,) + tup for tup in PRODUCT_EXOG_LTC]
TEST_CASES_EXOG_LTC = [("toy_model_exog_ltc",) + tup for tup in EXOG_LTC]

RANDOM_TEST_SET_LTC_AND_JOB_OFFER = np.random.choice(
    ALL_WEALTH_GRIDS, size=10, replace=False
)
PRODUCT_TWO_EXOG_PROCESSES = list(
    product(RANDOM_TEST_SET_LTC_AND_JOB_OFFER, list(range(8)))
)
EXOG_LTC_AND_JOB_OFFER = [
    (euler_rhs_exog_ltc_and_job_offer,) + tup for tup in PRODUCT_TWO_EXOG_PROCESSES
]
TEST_CASES_EXOG_LTC_AND_JOB_OFFER = [
    ("toy_model_exog_ltc_and_job_offer",) + tup for tup in EXOG_LTC_AND_JOB_OFFER
]


@pytest.mark.parametrize(
    "toy_model, euler_rhs, wealth_idx, state_idx",
    # TEST_CASES_EXOG_LTC + TEST_CASES_EXOG_LTC_AND_JOB_OFFER,
    TEST_CASES_EXOG_LTC_AND_JOB_OFFER,
    # TEST_CASES_EXOG_LTC,
)
def test_two_period(toy_model, euler_rhs, wealth_idx, state_idx, request):
    toy_model = request.getfixturevalue(toy_model)

    params = toy_model["params"]
    options = toy_model["options"]

    quad_points, quad_weights = roots_sh_legendre(
        options["model_params"]["quadrature_points_stochastic"]
    )
    quad_draws = norm.ppf(quad_points) * 1

    endog_grid_period = toy_model["endog_grid"]
    policy_period = toy_model["policy_left"]
    period_specific_state_objects = toy_model["period_specific_state_objects"]
    state_space = toy_model["state_space"]

    period = 0
    state_choices_period = period_specific_state_objects[period]["state_choice_mat"]
    state_choice_idxs_of_state = np.where(
        period_specific_state_objects[period]["idx_parent_states"] == state_idx
    )[0]

    if len(options["state_space"]["exogenous_processes"]) == 2:
        initial_conditions = {}
        initial_conditions["bad_health"] = state_space["ltc"][state_idx] == 1
        initial_conditions["job_offer"] = 1
    else:
        initial_conditions = {}
        initial_conditions["bad_health"] = state_space["ltc"][state_idx]

    for state_choice_idx in state_choice_idxs_of_state:
        endog_grid = endog_grid_period[state_choice_idx, wealth_idx + 1]
        policy = policy_period[state_choice_idx]
        choice = state_choices_period["choice"][state_choice_idx]

        if ~np.isnan(endog_grid) and endog_grid > 0:
            initial_conditions["wealth"] = endog_grid

            cons_calc = policy[wealth_idx + 1]
            diff = euler_rhs(
                initial_conditions,
                params,
                quad_draws,
                quad_weights,
                choice,
                cons_calc,
            ) - marginal_utility(consumption=cons_calc, params=params)

            assert_allclose(diff, 0, atol=1e-6)
