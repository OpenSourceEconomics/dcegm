"""This is a test for a simple two period model with exogenous processes.

We test DC-EGM against the closed form solution of the Euler equation.

"""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm

import dcegm.toy_models as toy_models
from dcegm.backward_induction import get_solve_func_for_model
from dcegm.pre_processing.setup_model import create_model_dict
from tests.utils.euler_equation_two_period import (
    euler_rhs_exog_ltc,
    euler_rhs_exog_ltc_and_job_offer,
)

RANDOM_TEST_WEALTH = np.random.choice(list(range(100)), size=10, replace=False)


@pytest.fixture(scope="session")
def toy_model_exog_ltc_and_job_offer():

    model_funcs = toy_models.load_example_model_functions(
        "with_stochastic_ltc_and_job_offer"
    )
    params, options = toy_models.load_example_params_model_specs_and_config(
        "with_stochastic_ltc_and_job_offer"
    )

    out = {}
    out["model"] = create_model_dict(
        options=options,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["utility_functions_final_period"],
        budget_constraint=model_funcs["budget_constraint"],
    )

    out["marginal_utility"] = model_funcs["utility_functions"]["marginal_utility"]

    (
        out["value"],
        out["policy"],
        out["endog_grid"],
    ) = get_solve_func_for_model(
        out["model"]
    )(params)

    out["params"] = params
    out["options"] = options
    out["euler"] = euler_rhs_exog_ltc_and_job_offer

    return out


@pytest.fixture(scope="session")
def toy_model_exog_ltc():

    model_funcs = toy_models.load_example_model_functions("with_stochastic_ltc")
    params, options = toy_models.load_example_params_model_specs_and_config(
        "with_stochastic_ltc"
    )

    out = {}
    out["model"] = create_model_dict(
        options=options,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["utility_functions_final_period"],
        budget_constraint=model_funcs["budget_constraint"],
    )
    out["marginal_utility"] = model_funcs["utility_functions"]["marginal_utility"]

    (
        out["value"],
        out["policy"],
        out["endog_grid"],
    ) = get_solve_func_for_model(
        out["model"]
    )(params)

    out["params"] = params
    out["options"] = options
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
        options["model_params"]["n_quad_points"]
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

    if len(options["state_space"]["stochastic_states"]) == 2:
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
            ) - toy_model["marginal_utility"](consumption=cons_calc, params=params)

            assert_allclose(diff, 0, atol=1e-6)
