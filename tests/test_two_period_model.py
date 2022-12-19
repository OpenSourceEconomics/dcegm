"""This module will have a test for our two period model."""
from itertools import product

import numpy as np
import pandas as pd
import pytest
from dcegm.solve import solve_dcegm
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model.final_period import solve_final_period
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_choice_set,
)


def flow_util(consumption, choice, params):
    rho = params.loc[("utility_function", "rho"), "value"]
    delta = params.loc[("utility_function", "delta"), "value"]
    u = consumption ** (1 - rho) / (1 - rho) - delta * (1 - choice)
    return u


def marginal_utility(consumption, params):
    rho = params.loc[("utility_function", "rho"), "value"]
    u_prime = consumption ** (-rho)
    return u_prime


def inverse_marginal_utility(marginal_utility, params):
    rho = params.loc[("utility_function", "rho"), "value"]
    return marginal_utility ** (-1 / rho)


def budget_dcegm(state, savings_grid, income_shock, params, options):  # noqa: 100
    interest_factor = 1 + params.loc[("assets", "interest_rate"), "value"]
    health_costs = params.loc[("assets", "ltc_cost"), "value"]
    wage = params.loc[("wage", "wage_avg"), "value"]
    resources = np.empty((income_shock.shape[0], savings_grid.shape[0]))
    for index_shock, shock in enumerate(income_shock):
        resources[index_shock, :] = (
            interest_factor * savings_grid
            + (wage + shock) * (1 - state[1])
            - state[-1] * health_costs
        )

    return resources.clip(min=0.5)


def transitions_dcegm(state, params):
    p = params.loc[("transition", "ltc_prob"), "value"]
    if state[-1] == 1:
        return np.array([0, 1])
    elif state[-1] == 0:
        return np.array([1 - p, p])


def budget(lagged_resources, lagged_consumption, lagged_choice, wage, health, params):
    interest_factor = 1 + params.loc[("assets", "interest_rate"), "value"]
    health_costs = params.loc[("assets", "ltc_cost"), "value"]
    resources = (
        interest_factor * (lagged_resources - lagged_consumption)
        + wage * (1 - lagged_choice)
        - health * health_costs
    ).clip(min=0.5)
    return resources


def wage(nu, params):
    wage = params.loc[("wage", "wage_avg"), "value"] + nu
    return wage


def ltc_prob(params, lag_health, health):
    p = params.loc[("transition", "ltc_prob"), "value"]
    if (lag_health == 0) and (health == 1):
        pi = p
    elif (lag_health == 0) and (health == 0):
        pi = 1 - p
    elif (lag_health == 1) and (health == 0):
        pi = 0
    elif (lag_health == 1) and (health == 1):
        pi = 1
    else:
        raise ValueError("Health state not defined.")
    return pi


def choice_probs(cons, d, params):
    v = flow_util(cons, d, params)
    v_0 = flow_util(cons, 0, params)
    v_1 = flow_util(cons, 1, params)
    choice_prob = np.exp(v) / (np.exp(v_0) + np.exp(v_1))
    return choice_prob


# expected marginal utility for one realization of the wage shock
def m_util_aux(init_cond, params, choice_1, nu, consumption):
    budget_1 = init_cond["wealth"]
    health_state_1 = init_cond["health"]

    weighted_marginal = 0
    for health_state_2 in [0, 1]:
        for choice_2 in [0, 1]:
            budget_2 = budget(
                budget_1,
                consumption,
                choice_1,
                wage(nu, params),
                health_state_2,
                params,
            )
            marginal_util = marginal_utility(budget_2, params)
            choice_prob = choice_probs(budget_2, choice_2, params)
            health_prob = ltc_prob(params, health_state_1, health_state_2)
            weighted_marginal += choice_prob * health_prob * marginal_util

    return weighted_marginal


def euler_rhs(init_cond, params, draws, weights, choice_1, consumption):
    beta = params.loc[("beta", "beta"), "value"]
    interest_factor = 1 + params.loc[("assets", "interest_rate"), "value"]

    rhs = 0
    for index_draw, draw in enumerate(draws):
        marg_util_draw = m_util_aux(init_cond, params, choice_1, draw, consumption)
        rhs += weights[index_draw] * marg_util_draw
    return rhs * beta * interest_factor


WEALTH_GRID_POINTS = 100


@pytest.fixture()
def input_data():
    index = pd.MultiIndex.from_tuples(
        [("utility_function", "rho"), ("utility_function", "delta")],
        names=["category", "name"],
    )
    params = pd.DataFrame(data=[0.5, 0.5], columns=["value"], index=index)
    params.loc[("assets", "interest_rate"), "value"] = 0.02
    params.loc[("assets", "ltc_cost"), "value"] = 5
    params.loc[("assets", "max_wealth"), "value"] = 50
    params.loc[("wage", "wage_avg"), "value"] = 8
    params.loc[("shocks", "sigma"), "value"] = 1
    params.loc[("shocks", "lambda"), "value"] = 1
    params.loc[("transition", "ltc_prob"), "value"] = 0.3
    params.loc[("beta", "beta"), "value"] = 0.95
    options = {
        "n_periods": 2,
        "n_discrete_choices": 2,
        "grid_points_wealth": WEALTH_GRID_POINTS,
        "quadrature_points_stochastic": 5,
        "n_exog_processes": 2,
    }
    state_space_functions = {
        "create_state_space": create_state_space,
        "get_state_specific_choice_set": get_state_specific_choice_set,
    }
    utility_functions = {
        "utility": flow_util,
        "inverse_marginal_utility": inverse_marginal_utility,
        "marginal_utility": marginal_utility,
    }

    policy_calculated, value_calculated = solve_dcegm(
        params,
        options,
        utility_functions,
        budget_constraint=budget_dcegm,
        solve_final_period=solve_final_period,
        state_space_functions=state_space_functions,
        user_transition_function=transitions_dcegm,
    )
    out = {}
    out["params"] = params
    out["policy"] = policy_calculated
    out["options"] = options

    return out


TEST_CASES = list(product(list(range(WEALTH_GRID_POINTS)), list(range(4))))


@pytest.mark.parametrize(
    "wealth_id, state_id",
    TEST_CASES,
)
def test_two_period(input_data, wealth_id, state_id):
    quad_points, quad_weights = roots_sh_legendre(5)
    quad_draws = norm.ppf(quad_points) * 1

    params = input_data["params"]
    state_space, indexer = create_state_space(input_data["options"])

    initial_cond = {}
    state = state_space[state_id, :]

    if state[1] == 1:
        choice_range = [1]
    else:
        choice_range = [0, 1]
    initial_cond["health"] = state[-1]

    for choice_in_period_1 in choice_range:
        calculated_policy_func = input_data["policy"][
            state_id, choice_in_period_1, :, :
        ]
        wealth = calculated_policy_func[0, wealth_id + 1]
        if ~np.isnan(wealth) and wealth > 0:
            initial_cond["wealth"] = wealth

            cons_calc = calculated_policy_func[1, wealth_id + 1]
            diff = euler_rhs(
                initial_cond,
                params,
                quad_draws,
                quad_weights,
                choice_in_period_1,
                cons_calc,
            ) - marginal_utility(cons_calc, params)

            np.testing.assert_allclose(diff, 0, atol=1e-8)
