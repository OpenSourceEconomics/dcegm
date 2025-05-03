"""This is a test for a simple two period model with exogenous processes.

We test DC-EGM against the closed form solution of the Euler equation.

"""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm

from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import get_solve_func_for_model
from tests.test_models.exog_ltc_and_job_offer import budget_dcegm_exog_ltc_and_job_offer
from tests.test_models.exog_ltc_model import budget_dcegm_exog_ltc
from tests.test_models.two_period_models.euler_equation import (
    euler_rhs_exog_ltc,
    euler_rhs_exog_ltc_and_job_offer,
)
from toy_models.cons_ret_model_dcegm_paper.state_space_objects import (
    create_state_space_function_dict,
)
from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
    marginal_utility_crra,
)

WEALTH_GRID_POINTS = 100
ALL_WEALTH_GRIDS = list(range(WEALTH_GRID_POINTS))
RANDOM_TEST_WEALTH = np.random.choice(ALL_WEALTH_GRIDS, size=10, replace=False)


@pytest.fixture(scope="session")
def toy_model_exog_ltc_and_job_offer():
    from tests.test_models.exog_ltc_and_job_offer import OPTIONS, PARAMS

    out = {}
    out["model"] = setup_model(
        options=OPTIONS,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_dcegm_exog_ltc_and_job_offer,
    )

    (
        out["value"],
        out["policy"],
        out["endog_grid"],
    ) = get_solve_func_for_model(
        out["model"]
    )(PARAMS)

    out["params"] = PARAMS
    out["options"] = OPTIONS
    out["euler"] = euler_rhs_exog_ltc_and_job_offer

    return out


@pytest.fixture(scope="session")
def toy_model_exog_ltc():
    from tests.test_models.exog_ltc_model import OPTIONS, PARAMS

    out = {}
    out["model"] = setup_model(
        options=OPTIONS,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_dcegm_exog_ltc,
    )

    (
        out["value"],
        out["policy"],
        out["endog_grid"],
    ) = get_solve_func_for_model(
        out["model"]
    )(PARAMS)

    out["params"] = PARAMS
    out["options"] = OPTIONS
    out["euler"] = euler_rhs_exog_ltc

    return out


@pytest.mark.parametrize(
    "model_name, wealth_idx, state_idx",
    product(
        ["exog_ltc", "exog_ltc_and_job_offer"],
        RANDOM_TEST_WEALTH,
        range(4),  # 4 endog states in period 0
    ),
)
def test_two_period(
    model_name,
    wealth_idx,
    state_idx,
    toy_model_exog_ltc,
    toy_model_exog_ltc_and_job_offer,
):

    if model_name == "exog_ltc":
        toy_model = toy_model_exog_ltc
    elif model_name == "exog_ltc_and_job_offer":
        toy_model = toy_model_exog_ltc_and_job_offer
    else:
        raise ValueError("Model not implemented")
    params = toy_model["params"]
    options = toy_model["options"]

    quad_points, quad_weights = roots_sh_legendre(
        options["model_params"]["n_quad_points_stochastic"]
    )
    quad_draws = norm.ppf(quad_points) * params["sigma"]

    endog_grid_period = toy_model["endog_grid"]
    policy_period = toy_model["policy"]

    model_structure = toy_model["model"]["model_structure"]
    state_space_dict = model_structure["state_space_dict"]

    state_choice_space = model_structure["state_choice_space"]
    state_choice_space_0 = state_choice_space[state_choice_space[:, 0] == 0]
    parent_states_of_state = np.where(
        model_structure["map_state_choice_to_parent_state"] == state_idx
    )[0]

    if len(options["state_space"]["exogenous_processes"]) == 2:
        initial_conditions = {}
        initial_conditions["bad_health"] = state_space_dict["ltc"][state_idx] == 1
        initial_conditions["job_offer"] = 1
    else:
        initial_conditions = {}
        initial_conditions["bad_health"] = state_space_dict["ltc"][state_idx]

    for state_choice_idx in parent_states_of_state:
        endog_grid = endog_grid_period[state_choice_idx, wealth_idx + 1]
        cons_calc = policy_period[state_choice_idx, wealth_idx + 1]
        choice = state_choice_space_0[state_choice_idx, -1]

        if ~np.isnan(endog_grid) and endog_grid > 0:
            initial_conditions["wealth"] = endog_grid

            diff = toy_model["euler"](
                initial_conditions,
                params,
                quad_draws,
                quad_weights,
                choice,
                cons_calc,
            ) - marginal_utility_crra(consumption=cons_calc, params=params)

            assert_allclose(diff, 0, atol=1e-6)
