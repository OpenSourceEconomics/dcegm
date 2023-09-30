from itertools import product

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from dcegm.pre_processing.process_functions import (
    determine_function_arguments_and_partial_options,
)
from dcegm.pre_processing.state_space import create_state_choice_space
from dcegm.pre_processing.state_space import create_state_space
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

from tests.two_period_models.ltc_and_job_offer.dcegm_code import (
    budget_dcegm_two_exog_processes,
)
from tests.two_period_models.ltc_and_job_offer.dcegm_code import func_exog_job_offer
from tests.two_period_models.ltc_and_job_offer.dcegm_code import func_exog_ltc
from tests.two_period_models.ltc_and_job_offer.dcegm_code import marginal_utility
from tests.two_period_models.ltc_and_job_offer.eueler_equation_code import (
    euler_rhs_two_exog_processes,
)

WEALTH_GRID_POINTS = 10

TEST_CASES_TWO_EXOG_PROCESSES = list(
    product(list(range(WEALTH_GRID_POINTS)), list(range(8)))
)


@pytest.fixture(scope="module")
def input_data_two_exog_processes(state_space_functions, utility_functions):
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
    input_data_two_exog_processes, wealth_idx, state_idx
):
    quad_points, quad_weights = roots_sh_legendre(5)
    quad_draws = norm.ppf(quad_points) * 1

    params = input_data_two_exog_processes["params"]
    keys = params.index.droplevel("category").tolist()
    values = params["value"].tolist()
    params = dict(zip(keys, values))
    (
        state_space,
        map_state_to_state_space_index,
        states_names_without_exog,
        exog_state_names,
        n_exog_states,
        exog_state_space,
    ) = create_state_space(input_data_two_exog_processes["options"])
    model_params_options = input_data_two_exog_processes["options"]["model_params"]

    choice_specific_choice_set_processedd = (
        determine_function_arguments_and_partial_options(
            func=get_state_specific_feasible_choice_set,
            options=model_params_options,
        )
    )

    (
        state_choice_space,
        _map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat,
    ) = create_state_choice_space(
        state_space_options=input_data_two_exog_processes["options"]["state_space"],
        state_space=state_space,
        state_space_names=states_names_without_exog + exog_state_names,
        map_state_to_state_space_index=map_state_to_state_space_index,
        get_state_specific_choice_set=choice_specific_choice_set_processedd,
    )
    initial_conditions = {}
    state = state_space[state_idx, :]
    reshape_state_choice_vec_to_mat[state_idx]

    feasible_choice_set = choice_specific_choice_set_processedd(
        lagged_choice=state[1],
    )

    endog_grid_period = input_data_two_exog_processes["result"][state[0]]["endog_grid"]
    policy_period = input_data_two_exog_processes["result"][state[0]]["policy_left"]

    initial_conditions["bad_health"] = state[-2] == 1
    initial_conditions["job_offer"] = 1  # working (no retirement) in period 0

    for choice_in_period_1 in feasible_choice_set:
        state_choice_idx = reshape_state_choice_vec_to_mat[
            state_idx, choice_in_period_1
        ]

        endog_grid = endog_grid_period[state_choice_idx, wealth_idx + 1]
        policy = policy_period[state_choice_idx]

        if ~np.isnan(endog_grid) and endog_grid > 0:
            initial_conditions["wealth"] = endog_grid

            cons_calc = policy[wealth_idx + 1]
            diff = euler_rhs_two_exog_processes(
                initial_conditions,
                params,
                quad_draws,
                quad_weights,
                choice_in_period_1,
                cons_calc,
            ) - marginal_utility(consumption=cons_calc, params=params)

            assert_allclose(diff, 0, atol=1e-6)
