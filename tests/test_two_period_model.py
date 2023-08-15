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
    get_state_specific_feasible_choice_set,
)


def flow_util(consumption, choice, params_dict):
    rho = params_dict["rho"]
    delta = params_dict["delta"]
    return consumption ** (1 - rho) / (1 - rho) - delta * (1 - choice)


def marginal_utility(consumption, params_dict):
    rho = params_dict["rho"]
    return consumption ** (-rho)


def inverse_marginal_utility(marginal_utility, params_dict):
    rho = params_dict["rho"]
    return marginal_utility ** (-1 / rho)


def budget_dcegm(state, saving, income_shock, params_dict, options):  # noqa: ARG001
    interest_factor = 1 + params_dict["interest_rate"]
    health_costs = params_dict["ltc_cost"]
    wage = params_dict["wage_avg"]
    resource = (
        interest_factor * saving
        + (wage + income_shock) * (1 - state[1])
        - state[-1] * health_costs
    )
    return jnp.maximum(resource, 0.5)


def transitions_dcegm(state, params_dict):
    p = params_dict["ltc_prob"]
    dead = state[-1] == 1
    return dead * jnp.array([0, 1]) + (1 - dead) * jnp.array([1 - p, p])


def budget(
    lagged_resources,
    lagged_consumption,
    lagged_choice,
    wage,
    health,
    params_dict,
):
    interest_factor = 1 + params_dict["interest_rate"]
    health_costs = params_dict["ltc_cost"]
    return (
        interest_factor * (lagged_resources - lagged_consumption)
        + wage * (1 - lagged_choice)
        - health * health_costs
    ).clip(min=0.5)


def wage(nu, params_dict):
    return params_dict["wage_avg"] + nu


def prob_long_term_care_patient(params_dict, lag_health, health):
    p = params_dict["ltc_prob"]
    if (lag_health == 0) and (health == 1):
        pi = p
    elif lag_health == health == 0:
        pi = 1 - p
    elif lag_health == 1 and health == 0:
        pi = 0
    elif lag_health == health == 1:
        pi = 1

    return pi


def choice_probs(cons, d, params_dict):
    v = flow_util(cons, d, params_dict)
    v_0 = flow_util(cons, 0, params_dict)
    v_1 = flow_util(cons, 1, params_dict)
    return np.exp(v) / (np.exp(v_0) + np.exp(v_1))


def m_util_aux(init_cond, params_dict, choice_1, nu, consumption):
    """Return the expected marginal utility for one realization of the wage shock."""
    budget_1 = init_cond["wealth"]
    health_state_1 = init_cond["health"]

    weighted_marginal = 0
    for health_state_2 in (0, 1):
        for choice_2 in (0, 1):
            budget_2 = budget(
                budget_1,
                consumption,
                choice_1,
                wage(nu, params_dict),
                health_state_2,
                params_dict,
            )
            marginal_util = marginal_utility(budget_2, params_dict)
            choice_prob = choice_probs(budget_2, choice_2, params_dict)
            health_prob = prob_long_term_care_patient(
                params_dict,
                health_state_1,
                health_state_2,
            )
            weighted_marginal += choice_prob * health_prob * marginal_util

    return weighted_marginal


def euler_rhs(init_cond, params_dict, draws, weights, choice_1, consumption):
    beta = params_dict["beta"]
    interest_factor = 1 + params_dict["interest_rate"]

    rhs = 0
    for index_draw, draw in enumerate(draws):
        marg_util_draw = m_util_aux(init_cond, params_dict, choice_1, draw, consumption)
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
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
    }
    utility_functions = {
        "utility": flow_util,
        "inverse_marginal_utility": inverse_marginal_utility,
        "marginal_utility": marginal_utility,
    }

    solve_dcegm(
        params,
        options,
        utility_functions,
        budget_constraint=budget_dcegm,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
        transition_function=transitions_dcegm,
    )

    out = {}
    out["params"] = params
    out["options"] = options

    return out


TEST_CASES = list(product(list(range(WEALTH_GRID_POINTS)), list(range(4))))


@pytest.mark.parametrize(
    ("wealth_idx", "state_idx"),
    TEST_CASES,
)
def test_two_period(input_data, wealth_idx, state_idx):
    quad_points, quad_weights = roots_sh_legendre(5)
    quad_draws = norm.ppf(quad_points) * 1

    params = input_data["params"]
    keys = params.index.droplevel("category").tolist()
    values = params["value"].tolist()
    params_dict = dict(zip(keys, values, strict=True))

    state_space, map_state_to_index = create_state_space(input_data["options"])
    (
        state_choice_space,
        _map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat,
        _transform_between_state_and_state_choice_space,
    ) = create_state_choice_space(
        state_space,
        map_state_to_index,
        get_state_specific_feasible_choice_set,
    )
    initial_cond = {}
    state = state_space[state_idx, :]
    idxs_state_choice_combs = reshape_state_choice_vec_to_mat[state_idx]
    initial_cond["health"] = state[-1]

    endog_grid_period = np.load(f"endog_grid_{state[0]}.npy")
    policy_period = np.load(f"policy_{state[0]}.npy")

    for state_choice_idx in idxs_state_choice_combs:
        choice_in_period_1 = state_choice_space[state_choice_idx][-1]

        endog_grid = endog_grid_period[state_choice_idx, wealth_idx + 1]
        policy = policy_period[state_choice_idx]

        if ~np.isnan(endog_grid) and endog_grid > 0:
            initial_cond["wealth"] = endog_grid

            cons_calc = policy[wealth_idx + 1]
            diff = euler_rhs(
                initial_cond,
                params_dict,
                quad_draws,
                quad_weights,
                choice_in_period_1,
                cons_calc,
            ) - marginal_utility(cons_calc, params_dict)

            assert_allclose(diff, 0, atol=1e-6)
