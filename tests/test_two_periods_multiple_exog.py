from itertools import product

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from dcegm.pre_processing.process_model import (
    process_model_functions_and_create_state_space_objects,
)
from dcegm.solve import solve_dcegm
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period_scalar,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.state_space_objects import update_state

from tests.two_period_models.ltc_and_job_offer.dcegm_code import (
    budget_dcegm_two_exog_processes,
)
from tests.two_period_models.ltc_and_job_offer.dcegm_code import flow_util
from tests.two_period_models.ltc_and_job_offer.dcegm_code import func_exog_job_offer
from tests.two_period_models.ltc_and_job_offer.dcegm_code import func_exog_ltc
from tests.two_period_models.ltc_and_job_offer.dcegm_code import (
    inverse_marginal_utility,
)
from tests.two_period_models.ltc_and_job_offer.dcegm_code import marginal_utility
from tests.two_period_models.ltc_and_job_offer.eueler_equation_code import (
    euler_rhs_two_exog_processes,
)

WEALTH_GRID_POINTS = 100

TEST_CASES_TWO_EXOG_PROCESSES = list(
    product(list(range(WEALTH_GRID_POINTS)), list(range(8)))
)


@pytest.fixture(scope="module")
def state_space_functions():
    out = {
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
        "update_endog_state_by_state_and_choice": update_state,
    }
    return out


@pytest.fixture(scope="module")
def utility_functions():
    out = {
        "utility": flow_util,
        "inverse_marginal_utility": inverse_marginal_utility,
        "marginal_utility": marginal_utility,
    }
    return out


@pytest.fixture(scope="module")
def input_data_two_exog_processes(state_space_functions, utility_functions):
    # ToDo: Write this as dictionary such that it has a much nicer overview
    index = pd.MultiIndex.from_tuples(
        [("utility_function", "rho"), ("utility_function", "delta")],
        names=["category", "name"],
    )
    params = pd.DataFrame(data=[0.5, 0.5], columns=["value"], index=index)
    params.loc[("assets", "interest_rate"), "value"] = 0.02
    params.loc[("assets", "ltc_cost"), "value"] = 5
    params.loc[("wage", "wage_avg"), "value"] = 8
    params.loc[("shocks", "sigma"), "value"] = 1
    params.loc[("shocks", "lambda"), "value"] = 1
    params.loc[("transition", "ltc_prob"), "value"] = 0.3
    params.loc[("beta", "beta"), "value"] = 0.95

    # exog params
    params.loc[("ltc_prob_constant", "ltc_prob_constant"), "value"] = 0.3
    params.loc[("ltc_prob_age", "ltc_prob_age"), "value"] = 0.1
    params.loc[("job_offer_constant", "job_offer_constant"), "value"] = 0.5
    params.loc[("job_offer_age", "job_offer_age"), "value"] = 0
    params.loc[("job_offer_educ", "job_offer_educ"), "value"] = 0
    params.loc[("job_offer_type_two", "job_offer_type_two"), "value"] = 0.4

    options = {
        "model_params": {
            "n_grid_points": WEALTH_GRID_POINTS,
            "n_choices": 2,
            "max_wealth": 50,
            "quadrature_points_stochastic": 5,
        },
        "state_space": {
            "n_periods": 2,
            "choices": np.arange(2),
            "endogenous_states": {
                "married": [0, 1],
            },
            "exogenous_processes": {
                "ltc": {"transition": func_exog_ltc, "states": [0, 1]},
                "job_offer": {"transition": func_exog_job_offer, "states": [0, 1]},
            },
        },
    }

    exog_savings_grid = jnp.linspace(
        0,
        options["model_params"]["max_wealth"],
        options["model_params"]["n_grid_points"],
    )

    result_dict = solve_dcegm(
        params,
        options,
        exog_savings_grid=exog_savings_grid,
        utility_functions=utility_functions,
        budget_constraint=budget_dcegm_two_exog_processes,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
    )

    out = {}
    out["params"] = params
    out["options"] = options
    out["result"] = result_dict

    return out


@pytest.mark.parametrize(
    "wealth_idx, state_idx",
    TEST_CASES_TWO_EXOG_PROCESSES,
)
def test_two_period_two_exog_processes(
    input_data_two_exog_processes,
    wealth_idx,
    state_idx,
    utility_functions,
    state_space_functions,
):
    quad_points, quad_weights = roots_sh_legendre(5)
    quad_draws = norm.ppf(quad_points) * 1

    params = input_data_two_exog_processes["params"]
    keys = params.index.droplevel("category").tolist()
    values = params["value"].tolist()
    params = dict(zip(keys, values))
    (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_beginning_of_period_wealth,
        compute_final_period,
        compute_exog_transition_vec,
        compute_upper_envelope,
        period_specific_state_objects,
        state_space,
    ) = process_model_functions_and_create_state_space_objects(
        options=input_data_two_exog_processes["options"],
        user_utility_functions=utility_functions,
        user_budget_constraint=budget_dcegm_two_exog_processes,
        user_final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
    )
    period = state_space["period"][state_idx]

    endog_grid_period = input_data_two_exog_processes["result"][period]["endog_grid"]
    policy_period = input_data_two_exog_processes["result"][period]["policy_left"]

    state_choices_period = period_specific_state_objects[period]["state_choice_mat"]

    state_choice_idxs_of_state = np.where(
        period_specific_state_objects[period]["idx_parent_states"] == state_idx
    )[0]

    initial_conditions = {}
    initial_conditions["bad_health"] = state_space["ltc"][state_idx] == 1
    initial_conditions["job_offer"] = 1  # working (no retirement) in period 0

    for state_choice_idx in state_choice_idxs_of_state:
        endog_grid = endog_grid_period[state_choice_idx, wealth_idx + 1]
        policy = policy_period[state_choice_idx]
        choice = state_choices_period["choice"][state_choice_idx]

        if ~np.isnan(endog_grid) and endog_grid > 0:
            initial_conditions["wealth"] = endog_grid

            cons_calc = policy[wealth_idx + 1]
            diff = euler_rhs_two_exog_processes(
                initial_conditions,
                params,
                quad_draws,
                quad_weights,
                choice,
                cons_calc,
            ) - marginal_utility(consumption=cons_calc, params=params)

            assert_allclose(diff, 0, atol=1e-6)
