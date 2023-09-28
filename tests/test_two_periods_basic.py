"""This is a test for a simple two period model with exogenous processes.

We test DC-EGM against the closed form solution of the Euler equation.

"""
from itertools import product

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from dcegm.solve import solve_dcegm
from dcegm.state_space import create_state_choice_space
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period_scalar,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    update_state,
)


def flow_util(consumption, choice, params):
    return consumption ** (1 - params["rho"]) / (1 - params["rho"]) - params[
        "delta"
    ] * (1 - choice)


def marginal_utility(consumption, params):
    return consumption ** (-params["rho"])


def inverse_marginal_utility(marginal_utility, params):
    return marginal_utility ** (-1 / params["rho"])


def func_exog_ltc(
    period,
    lagged_ltc,
    params,
):
    prob_ltc = (lagged_ltc == 0) * (
        params["ltc_prob_constant"] + period * params["ltc_prob_age"]
    ) + (lagged_ltc == 1)
    prob_no_ltc = 1 - prob_ltc

    return prob_no_ltc, prob_ltc


def budget_dcegm(state, saving, income_shock, options, params):
    ltc_patient = state[-1] == 1

    resource = (
        (1 + params["interest_rate"]) * saving
        + (params["wage_avg"] + income_shock) * (1 - state[1])  # if worked last period
        - ltc_patient * params["ltc_cost"]
    )
    return jnp.maximum(resource, 0.5)


def budget(
    lagged_resources,
    lagged_consumption,
    lagged_retirement_choice,
    wage,
    bad_health,
    params,
):
    interest_factor = 1 + params["interest_rate"]
    health_costs = params["ltc_cost"]
    resources = (
        interest_factor * (lagged_resources - lagged_consumption)
        + wage * (1 - lagged_retirement_choice)
        - bad_health * health_costs
    ).clip(min=0.5)
    return resources


def wage(nu, params):
    wage = params["wage_avg"] + nu
    return wage


def prob_long_term_care_patient(params, lagged_bad_health, bad_health):
    p = params["ltc_prob"]
    # jnp.array([[0.7, 0.3], [0, 1]])

    if lagged_bad_health == bad_health == 0:
        pi = 1 - p
    elif (lagged_bad_health == 0) and (bad_health == 1):
        pi = p
    elif lagged_bad_health == 1 and bad_health == 0:
        pi = 0
    elif lagged_bad_health == bad_health == 1:
        pi = 1

    return pi


def choice_prob_retirement(consumption, choice, params):
    v = flow_util(consumption=consumption, params=params, choice=choice)
    v_0 = flow_util(consumption=consumption, params=params, choice=0)
    v_1 = flow_util(consumption=consumption, params=params, choice=1)
    choice_prob = np.exp(v) / (np.exp(v_0) + np.exp(v_1))
    return choice_prob


def marginal_utility_weighted(init_cond, params, retirement_choice_1, nu, consumption):
    """Return the expected marginal utility for one realization of the wage shock."""
    budget_1 = init_cond["wealth"]
    ltc_state_1 = init_cond["bad_health"]

    weighted_marginal = 0
    for ltc_state_2 in (0, 1):
        for retirement_choice_2 in (0, 1):
            budget_2 = budget(
                budget_1,
                consumption,
                retirement_choice_1,
                wage(nu, params),
                ltc_state_2,
                params,
            )

            marginal_util = marginal_utility(consumption=budget_2, params=params)
            choice_prob = choice_prob_retirement(
                consumption=budget_2, choice=retirement_choice_2, params=params
            )

            ltc_prob = prob_long_term_care_patient(params, ltc_state_1, ltc_state_2)

            weighted_marginal += choice_prob * ltc_prob * marginal_util

    return weighted_marginal


def euler_rhs(init_cond, params, draws, weights, retirement_choice_1, consumption):
    beta = params["beta"]
    interest_factor = 1 + params["interest_rate"]

    rhs = 0
    for index_draw, draw in enumerate(draws):
        marg_util_draw = marginal_utility_weighted(
            init_cond, params, retirement_choice_1, draw, consumption
        )
        rhs += weights[index_draw] * marg_util_draw
    return rhs * beta * interest_factor


WEALTH_GRID_POINTS = 100


@pytest.fixture(scope="module")
def input_data():
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

    options = {
        "model_params": {
            "n_grid_points": WEALTH_GRID_POINTS,
            "max_wealth": 50,
            "quadrature_points_stochastic": 5,
        },
        "state_space": {
            "endogenous_states": {
                "period": np.arange(2),
                "lagged_choice": [0, 1],
            },
            "exogenous_states": {"lagged_ltc": [0, 1]},
            "exogenous_processes": {
                "ltc": func_exog_ltc,
            },
            "choice": [0, 1],
        },
    }
    state_space_functions = {
        "create_state_space": create_state_space,
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
        "update_endog_state_by_state_and_choice": update_state,
    }
    utility_functions = {
        "utility": flow_util,
        "inverse_marginal_utility": inverse_marginal_utility,
        "marginal_utility": marginal_utility,
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
        budget_constraint=budget_dcegm,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
    )

    out = {}
    out["params"] = params
    out["options"] = options
    out["result"] = result_dict

    return out


TEST_CASES = list(product(list(range(WEALTH_GRID_POINTS)), list(range(4))))


@pytest.mark.parametrize(
    "wealth_idx, state_idx",
    TEST_CASES,
)
def test_two_period(input_data, wealth_idx, state_idx):
    quad_points, quad_weights = roots_sh_legendre(5)
    quad_draws = norm.ppf(quad_points) * 1

    params = input_data["params"]
    keys = params.index.droplevel("category").tolist()
    values = params["value"].tolist()
    params = dict(zip(keys, values))
    (
        state_space,
        map_state_to_index,
    ) = create_state_space(input_data["options"]["state_space"])
    (
        state_choice_space,
        _map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat,
    ) = create_state_choice_space(
        state_space_options=input_data["options"]["state_space"],
        state_space=state_space,
        map_state_to_state_space_index=map_state_to_index,
        get_state_specific_choice_set=get_state_specific_feasible_choice_set,
    )
    initial_conditions = {}
    state = state_space[state_idx, :]
    reshape_state_choice_vec_to_mat[state_idx]

    initial_conditions["bad_health"] = state[-1]

    feasible_choice_set = get_state_specific_feasible_choice_set(
        state, map_state_to_index
    )

    endog_grid_period = input_data["result"][state[0]]["endog_grid"]
    policy_period = input_data["result"][state[0]]["policy_left"]

    for choice_in_period_1 in feasible_choice_set:
        state_choice_idx = reshape_state_choice_vec_to_mat[
            state_idx, choice_in_period_1
        ]

        endog_grid = endog_grid_period[state_choice_idx, wealth_idx + 1]
        policy = policy_period[state_choice_idx]

        if ~np.isnan(endog_grid) and endog_grid > 0:
            initial_conditions["wealth"] = endog_grid

            cons_calc = policy[wealth_idx + 1]
            diff = euler_rhs(
                initial_conditions,
                params,
                quad_draws,
                quad_weights,
                choice_in_period_1,
                cons_calc,
            ) - marginal_utility(consumption=cons_calc, params=params)

            assert_allclose(diff, 0, atol=1e-6)


# ======================================================================================
# Two Exogenous Processes
# ======================================================================================
