"""This is a test for a simple two period model with exogenous processes.

We test DC-EGM against the closed form solution of the Euler equation.

"""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm

import dcegm.toy_models.exogenous_ltc.budget_equation
import dcegm.toy_models.exogenous_ltc.params_and_options
import dcegm.toy_models.exogenous_ltc_and_job_offer.params_and_options
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import get_solve_func_for_model
from dcegm.toy_models.cons_ret_model_dcegm_paper import (
    create_final_period_utility_function_dict,
    create_state_space_function_dict,
    create_utility_function_dict,
    marginal_utility_crra,
)
from tests.utils.euler_equation_two_perio import (
    euler_rhs_exog_ltc,
    euler_rhs_exog_ltc_and_job_offer,
)

RANDOM_TEST_WEALTH = np.random.choice(list(range(100)), size=10, replace=False)


@pytest.fixture(scope="session")
def toy_model_exog_ltc_and_job_offer():

    out = {}
    out["model"] = setup_model(
        options=dcegm.toy_models.exogenous_ltc_and_job_offer.params_and_options.OPTIONS,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=ltc_and_job_offer_file.budget_dcegm_exog_ltc_and_job_offer,
    )

    (
        out["value"],
        out["policy"],
        out["endog_grid"],
    ) = get_solve_func_for_model(
        out["model"]
    )(dcegm.toy_models.exogenous_ltc_and_job_offer.params_and_options.PARAMS)

    out["params"] = (
        dcegm.toy_models.exogenous_ltc_and_job_offer.params_and_options.PARAMS
    )
    out["options"] = (
        dcegm.toy_models.exogenous_ltc_and_job_offer.params_and_options.OPTIONS
    )
    out["euler"] = euler_rhs_exog_ltc_and_job_offer

    return out


@pytest.fixture(scope="session")
def toy_model_exog_ltc():

    out = {}
    out["model"] = setup_model(
        options=dcegm.toy_models.exogenous_ltc.params_and_options.OPTIONS,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=dcegm.toy_models.exogenous_ltc.budget_equation.budget_equation_with_ltc,
    )

    (
        out["value"],
        out["policy"],
        out["endog_grid"],
    ) = get_solve_func_for_model(
        out["model"]
    )(dcegm.toy_models.exogenous_ltc.params_and_options.PARAMS)

    out["params"] = dcegm.toy_models.exogenous_ltc.params_and_options.PARAMS
    out["options"] = dcegm.toy_models.exogenous_ltc.params_and_options.OPTIONS
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
