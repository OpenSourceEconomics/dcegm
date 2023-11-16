from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.solve import solve_dcegm
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.state_space_objects import update_state
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_final_consume_all,
)
from toy_models.consumption_retirement_model.utility_functions import (
    utility_final_consume_all,
)

from tests.two_period_models.exog_ltc_and_job_offer.euler_equation import (
    euler_rhs_two_exog_processes,
)
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    budget_dcegm_two_exog_processes,
)
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import flow_util
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    func_exog_job_offer,
)
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import func_exog_ltc
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    inverse_marginal_utility,
)
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    marginal_utility,
)

WEALTH_GRID_POINTS = 100
ALL_WEALTH_GRIDS = list(range(WEALTH_GRID_POINTS))
RANDOM_TEST_SET = np.random.choice(ALL_WEALTH_GRIDS, size=10, replace=False)
TEST_CASES_TWO_EXOG_PROCESSES = list(product(RANDOM_TEST_SET, list(range(8))))


@pytest.fixture(scope="module")
def state_space_functions():
    """Return dict with state space functions."""
    out = {
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
        "update_endog_state_by_state_and_choice": update_state,
    }
    return out


@pytest.fixture(scope="module")
def utility_functions():
    """Return dict with utility functions."""
    out = {
        "utility": flow_util,
        "inverse_marginal_utility": inverse_marginal_utility,
        "marginal_utility": marginal_utility,
    }
    return out


@pytest.fixture(scope="module")
def utility_functions_final_period():
    """Return dict with utility functions for final period."""
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


@pytest.fixture(scope="module")
def input_data(
    state_space_functions, utility_functions, utility_functions_final_period
):
    # ToDo: Write this as dictionary such that it has a much nicer overview
    params = {}
    params["rho"] = 0.5
    params["delta"] = 0.5
    params["interest_rate"] = 0.02
    params["ltc_cost"] = 5
    params["wage_avg"] = 8
    params["sigma"] = 1
    params["lambda"] = 1
    params["ltc_prob"] = 0.3
    params["beta"] = 0.95

    # exog params
    params["ltc_prob_constant"] = 0.3
    params["ltc_prob_age"] = 0.1
    params["job_offer_constant"] = 0.5
    params["job_offer_age"] = 0
    params["job_offer_educ"] = 0
    params["job_offer_type_two"] = 0.4

    options = {
        "model_params": {
            "n_choices": 2,
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
        50,
        WEALTH_GRID_POINTS,
    )

    (
        _model_funcs,
        _compute_upper_envelope,
        get_state_specific_choice_set,
        update_endog_state_by_state_and_choice,
    ) = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm_two_exog_processes,
    )
    out = {}

    (
        out["period_specific_state_objects"],
        out["state_space"],
        _,
        _,
    ) = create_state_space_and_choice_objects(
        options=options,
        get_state_specific_choice_set=get_state_specific_choice_set,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    (
        out["value"],
        out["policy_left"],
        out["policy_right"],
        out["endog_grid"],
    ) = solve_dcegm(
        params,
        options,
        exog_savings_grid=exog_savings_grid,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm_two_exog_processes,
    )

    out["params"] = params
    out["options"] = options

    return out


@pytest.mark.parametrize(
    "wealth_idx, state_idx",
    TEST_CASES_TWO_EXOG_PROCESSES,
)
def test_two_period_two_exog_processes(
    input_data,
    wealth_idx,
    state_idx,
    utility_functions,
    state_space_functions,
):
    quad_points, quad_weights = roots_sh_legendre(5)
    quad_draws = norm.ppf(quad_points) * 1

    params = input_data["params"]

    period = 0

    endog_grid_period = input_data["endog_grid"]
    policy_period = input_data["policy_left"]
    period_specific_state_objects = input_data["period_specific_state_objects"]
    state_space = input_data["state_space"]

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
