"""This is a test for a simple two period model with exogenous processes.

We test DC-EGM against the closed form solution of the Euler equation.

"""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm

import dcegm
import dcegm.toy_models as toy_models
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
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "with_stochastic_ltc_and_job_offer"
        )
    )
    out = {}
    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_funcs,
    )

    out["marginal_utility"] = model_funcs["utility_functions"]["marginal_utility"]

    out["model_solved"] = model.solve(params)

    out["params"] = params
    out["model_specs"] = model_specs
    out["model_config"] = model_config
    out["euler"] = euler_rhs_exog_ltc_and_job_offer

    return out


@pytest.fixture(scope="session")
def toy_model_exog_ltc():

    model_funcs = toy_models.load_example_model_functions("with_stochastic_ltc")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_stochastic_ltc")
    )

    out = {}
    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_funcs,
    )

    out["model_solved"] = model.solve(params)
    out["marginal_utility"] = model_funcs["utility_functions"]["marginal_utility"]

    out["params"] = params
    out["model_config"] = model_config
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

    quad_points, quad_weights = roots_sh_legendre(
        toy_model["model_config"]["n_quad_points"]
    )
    quad_draws = norm.ppf(quad_points) * params["income_shock_std"]

    endog_grid_period = toy_model["model_solved"].endog_grid
    policy_period = toy_model["model_solved"].policy

    model_structure = toy_model["model_solved"].model_structure
    state_space_dict = model_structure["state_space_dict"]

    state_choice_space = model_structure["state_choice_space"]
    state_choice_space_0 = state_choice_space[state_choice_space[:, 0] == 0]
    parent_states_of_state = np.where(
        model_structure["map_state_choice_to_parent_state"] == state_idx
    )[0]

    if len(toy_model["model_config"]["stochastic_states"]) == 2:
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
            initial_conditions["assets_end_of_period"] = endog_grid

            rhs_euler = toy_model["euler"](
                initial_conditions,
                params,
                quad_draws,
                quad_weights,
                choice,
                cons_calc,
            )

            lhs = toy_model["marginal_utility"](consumption=cons_calc, params=params)

            diff = lhs - rhs_euler

            assert_allclose(diff, 0, atol=1e-6)
