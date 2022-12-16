"""This module will have a test for our two period model."""
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.special import roots_hermite


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
    resources = (
        interest_factor * savings_grid
        + (wage + income_shock) * (1 - state[1])
        - state[-1] * health_costs
    )
    return resources


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
    )
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
        rhs += weights[index_draw] * beta * interest_factor * marg_util_draw
    return rhs


quad_draws_unscaled, quad_weights = roots_hermite(5)
quad_weights *= 1 / np.sqrt(np.pi)
quad_draws = quad_draws_unscaled * np.sqrt(2) * 1


def diff_func(partial_lhs, partial_rhs, cons):
    return partial_lhs(cons) - partial_rhs(cons)


index = pd.MultiIndex.from_tuples(
    [("utility_function", "rho"), ("utility_function", "delta")],
    names=["category", "name"],
)
params = pd.DataFrame(data=[0.5, 0.5], columns=["value"], index=index)
params.loc[("assets", "interest_rate"), "value"] = 0.02
params.loc[("assets", "ltc_cost"), "value"] = 5
params.loc[("wage", "wage_avg"), "value"] = 8
params.loc[("transition", "ltc_prob"), "value"] = 0.3
params.loc[("beta", "beta"), "value"] = 0.95

initial_cond = {"health": 0, "wealth": 10}
choice_in_period_1 = 0

partial_euler = partial(
    euler_rhs, initial_cond, params, quad_draws, quad_weights, choice_in_period_1
)
partial_marginal = partial(marginal_utility, params=params)
partil_diff = partial(diff_func, partial_euler, partial_marginal)

root_scalar(partil_diff, method="brenth", bracket=[0.001, initial_cond["wealth"]])
