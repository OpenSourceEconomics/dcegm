"""This is a test for a simple two period model with exogenous processes.

We test DC-EGM against the closed form solution of the Euler equation.

"""
from functools import partial
from itertools import product
from typing import Callable
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from dcegm.solve import solve_dcegm
from dcegm.state_space import create_state_choice_space
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space_two_exog_processes,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_feasible_choice_set_two_exog_processes,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    update_state,
)


def flow_util(consumption, choice, rho, delta):
    return consumption ** (1 - rho) / (1 - rho) - delta * (1 - choice)


def marginal_utility(consumption, rho):
    return consumption ** (-rho)


def inverse_marginal_utility(marginal_utility, rho):
    return marginal_utility ** (-1 / rho)


def solve_final_period_scalar(
    state_vec: np.ndarray,  # noqa: U100
    choice: int,
    begin_of_period_resources: float,
    rho: float,
    delta: float,
    compute_utility: Callable,
    compute_marginal_utility: Callable,
) -> Tuple[float, float, float]:
    """Compute optimal consumption policy and value function in the final period.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) containing the
            period-specific state vector.
        choice (int): The agent's choice in the current period.
        begin_of_period_resources (float): The agent's begin of period resources.
        compute_utility (callable): Function for computation of agent's utility.
        compute_marginal_utility (callable): Function for computation of agent's
        params (dict): Dictionary of model parameters.
        options (dict): Options dictionary.

    Returns:
        tuple:

        - consumption (float): The agent's consumption in the final period.
        - value (float): The agent's value in the final period.
        - marginal_utility (float): The agent's marginal utility .

    """
    consumption = begin_of_period_resources

    value = compute_utility(
        consumption=begin_of_period_resources, choice=choice, rho=rho, delta=delta
    )

    marginal_utility = compute_marginal_utility(
        consumption=begin_of_period_resources, rho=rho
    )

    return marginal_utility, value, consumption


def get_transition_vector_dcegm(state, params, transition_matrix):
    return transition_matrix[state[-1]]


def get_transition_vector_dcegm_two_exog_processes(state, params, transition_matrix):
    # state[-1] is the exogenous state (combined health and job offer)
    # state[0] is the endogenous state variable period
    # i.e. we allow for period (age) specific transition probabilities
    return transition_matrix[state[-1], ..., state[0]]


def budget_dcegm(
    state, savings, income_shock, max_wealth, interest_rate, ltc_cost, wage_avg
):
    resource = (
        (1 + interest_rate) * savings
        + (wage_avg + income_shock) * (1 - state[1])
        - state[-1] * ltc_cost
    )
    return jnp.maximum(resource, 0.5)


def budget_dcegm_two_exog_processes(
    state, saving, income_shock, max_wealth, interest_rate, ltc_cost, wage_avg
):  # noqa: U100
    # lagged_job_offer = jnp.abs(state[-1] - 2) * (state[-1] > 0) * state[0]  # [1, 3]
    ltc_patient = state[-1] > 1  # [2, 3]

    resource = (
        (1 + interest_rate) * saving
        + (wage_avg + income_shock) * (1 - state[1])  # if worked last period
        - ltc_patient * ltc_cost
    )
    return jnp.maximum(resource, 0.5)


def budget(
    lagged_resources,
    lagged_consumption,
    lagged_retirement_choice,
    wage,
    health,
    params,
):
    interest_factor = 1 + params["interest_rate"]
    health_costs = params["ltc_cost"]
    resources = (
        interest_factor * (lagged_resources - lagged_consumption)
        + wage * (1 - lagged_retirement_choice)
        - health * health_costs
    ).clip(min=0.5)
    return resources


def budget_two_exog_processes(
    lagged_resources,
    lagged_consumption,
    lagged_retirement_choice,
    wage,
    bad_health,
    lagged_job_offer,
    params,
):
    interest_factor = 1 + params["interest_rate"]
    health_costs = params["ltc_cost"]
    resources = (
        interest_factor * (lagged_resources - lagged_consumption)
        + wage * lagged_job_offer * (1 - lagged_retirement_choice)
        - bad_health * health_costs
    ).clip(min=0.5)
    return resources


def wage(nu, params):
    wage = params["wage_avg"] + nu
    return wage


def prob_long_term_care_patient(params, lagged_bad_health, bad_health):
    p = params["ltc_prob"]

    if (lagged_bad_health == 0) and (bad_health == 1):
        pi = p
    elif lagged_bad_health == bad_health == 0:
        pi = 1 - p
    elif lagged_bad_health == 1 and bad_health == 0:
        pi = 0
    elif lagged_bad_health == bad_health == 1:
        pi = 1

    return pi


def prob_job_offer(params, lagged_job_offer, job_offer):
    # p = params["job_offer_prob"]

    if (lagged_job_offer == 0) and (job_offer == 1):
        pi = 0.5
    elif lagged_job_offer == job_offer == 0:
        pi = 0.5
    elif lagged_job_offer == 1 and job_offer == 0:
        pi = 0.1
    elif lagged_job_offer == job_offer == 1:
        pi = 0.9

    return pi


def choice_prob_retirement(consumption, choice, rho, delta):
    v = flow_util(consumption=consumption, choice=choice, rho=rho, delta=delta)
    v_0 = flow_util(consumption=consumption, choice=0, rho=rho, delta=delta)
    v_1 = flow_util(consumption=consumption, choice=1, rho=rho, delta=delta)
    choice_prob = np.exp(v) / (np.exp(v_0) + np.exp(v_1))
    return choice_prob


def m_util_aux(
    state,
    init_cond,
    params,
    retirement_choice_1,
    nu,
    consumption,
    get_transition_vector_by_state,
):
    """Return the expected marginal utility for one realization of the wage shock."""
    budget_1 = init_cond["wealth"]
    # ltc_state_1 = init_cond["bad_health"]

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
            marginal_util = marginal_utility(consumption=budget_2, rho=params["rho"])
            choice_prob = choice_prob_retirement(
                consumption=budget_2,
                choice=retirement_choice_2,
                rho=params["rho"],
                delta=params["delta"],
            )
            # ltc_prob = prob_long_term_care_patient(
            #     params, ltc_state_1, ltc_state_2
            # )
            ltc_prob = get_transition_vector_by_state[ltc_state_2]

            weighted_marginal += choice_prob * ltc_prob * marginal_util

    return weighted_marginal


def marginal_utility_weighted_two_exog_processes(
    init_cond, params, retirement_choice_1, nu, consumption
):
    """Return the expected marginal utility for one realization of the wage shock."""
    budget_1 = init_cond["wealth"]
    ltc_state_1 = init_cond["bad_health"]
    job_state_1 = init_cond["job_offer"]

    weighted_marginal = 0
    for ltc_state_2 in (0, 1):
        for job_state_2 in (0, 1):
            for retirement_choice_2 in (0, 1):
                budget_2 = budget_two_exog_processes(
                    budget_1,
                    consumption,
                    retirement_choice_1,
                    wage(nu, params),
                    ltc_state_2,
                    job_state_1,
                    params,
                )

                # marginal_util = marginal_utility(budget_2, params)
                # choice_prob = choice_prob_retirement(
                #     budget_2, retirement_choice_2, params
                # )
                marginal_util = marginal_utility(
                    consumption=budget_2, rho=params["rho"]
                )
                choice_prob = choice_prob_retirement(
                    consumption=budget_2,
                    choice=retirement_choice_2,
                    rho=params["rho"],
                    delta=params["delta"],
                )

                ltc_prob = prob_long_term_care_patient(params, ltc_state_1, ltc_state_2)
                job_offer_prob = prob_job_offer(params, job_state_1, job_state_2)

                weighted_marginal += (
                    choice_prob * ltc_prob * job_offer_prob * marginal_util
                )

    return weighted_marginal


def euler_rhs(
    state,
    init_cond,
    params,
    draws,
    weights,
    retirement_choice_1,
    consumption,
    get_transition_vector_by_state,
):
    beta = params["beta"]
    interest_factor = 1 + params["interest_rate"]

    rhs = 0
    for index_draw, draw in enumerate(draws):
        marg_util_draw = m_util_aux(
            state,
            init_cond,
            params,
            retirement_choice_1,
            draw,
            consumption,
            get_transition_vector_by_state=get_transition_vector_by_state,
        )
        rhs += weights[index_draw] * marg_util_draw
    return rhs * beta * interest_factor


def euler_rhs_two_exog_processes(
    init_cond, params, draws, weights, retirement_choice_1, consumption
):
    beta = params["beta"]
    interest_factor = 1 + params["interest_rate"]

    rhs = 0
    for index_draw, draw in enumerate(draws):
        marg_util_draw = marginal_utility_weighted_two_exog_processes(
            init_cond, params, retirement_choice_1, draw, consumption
        )
        rhs += weights[index_draw] * marg_util_draw
    return rhs * beta * interest_factor


WEALTH_GRID_POINTS = 100


@pytest.fixture(scope="module")
def input_data():
    n_exog_states = 2

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
    options = {
        "n_periods": 2,
        "n_discrete_choices": 2,
        "n_grid_points": WEALTH_GRID_POINTS,
        "max_wealth": 50,
        "quadrature_points_stochastic": 5,
        "n_exog_states": n_exog_states,
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

    p = params.loc[("transition", "ltc_prob"), "value"]
    transition_matrix = jnp.array([[1 - p, p], [0, 1]])
    get_transition_vector_partial = partial(
        get_transition_vector_dcegm,
        transition_matrix=transition_matrix,
    )

    exog_savings_grid = np.linspace(0, options["max_wealth"], options["n_grid_points"])

    result_dict = solve_dcegm(
        params,
        options,
        exog_savings_grid=exog_savings_grid,
        utility_functions=utility_functions,
        budget_constraint=budget_dcegm,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
        transition_function=get_transition_vector_partial,
    )

    out = {}
    out["params"] = params
    out["options"] = options
    out["get_transition_vector_by_state"] = get_transition_vector_partial
    out["endog_grid"] = result_dict[0]["endog_grid"]
    out["policy_left"] = result_dict[0]["policy_left"]

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
    initial_conditions = {}
    state = state_space[state_idx, :]
    trans_vec = input_data["get_transition_vector_by_state"](state, params)

    idxs_state_choice_combs = reshape_state_choice_vec_to_mat[state_idx]
    initial_conditions["bad_health"] = state[-1]

    endog_grid_period = input_data["endog_grid"]
    policy_period = input_data["policy_left"]

    for state_choice_idx in idxs_state_choice_combs:
        choice_in_period_1 = state_choice_space[state_choice_idx][-1]

        endog_grid = endog_grid_period[state_choice_idx, wealth_idx + 1]
        policy = policy_period[state_choice_idx]

        if ~np.isnan(endog_grid) and endog_grid > 0:
            initial_conditions["wealth"] = endog_grid

            consumption = policy[wealth_idx + 1]
            diff = euler_rhs(
                state,
                initial_conditions,
                params,
                quad_draws,
                quad_weights,
                choice_in_period_1,
                consumption,
                get_transition_vector_by_state=trans_vec,
            ) - marginal_utility(consumption=consumption, rho=params["rho"])

            assert_allclose(diff, 0, atol=1e-6)


# ======================================================================================
# Two Exogenous Processes
# ======================================================================================


@pytest.fixture(scope="module")
def input_data_two_exog_processes():
    n_exog_states = 4

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
    # params.loc[("beta", "beta"), "value"] = 0.95
    options = {
        "n_periods": 2,
        "n_discrete_choices": 2,
        "n_grid_points": WEALTH_GRID_POINTS,
        "max_wealth": 50,
        "quadrature_points_stochastic": 5,
        "n_exog_states": n_exog_states,
    }
    state_space_functions = {
        "create_state_space": create_state_space_two_exog_processes,
        "get_state_specific_choice_set": get_feasible_choice_set_two_exog_processes,
        "update_endog_state_by_state_and_choice": update_state,
    }
    utility_functions = {
        "utility": flow_util,
        "inverse_marginal_utility": inverse_marginal_utility,
        "marginal_utility": marginal_utility,
    }

    ltc_probabilities = jnp.array([[0.7, 0.3], [0, 1]])
    job_offer_probabilities = jnp.array([[0.5, 0.5], [0.1, 0.9]])
    job_offer_probabilities_period_specific = jnp.repeat(
        job_offer_probabilities[:, :, jnp.newaxis], 2, axis=2
    )

    transition_matrix_age0 = jnp.kron(
        ltc_probabilities, job_offer_probabilities_period_specific[..., 0]
    )
    transition_matrix_age1 = jnp.kron(
        ltc_probabilities, job_offer_probabilities_period_specific[..., 1]
    )
    transition_matrix = jnp.dstack((transition_matrix_age0, transition_matrix_age1))
    # Has shape (4, 4, 4)
    # The third dimension, contains the age-specific transition matrices.
    # Note that age (or period) is a state variable.
    # [..., 0] is the transition matrix for age 0
    # [..., 1] is the transition matrix for age 1

    get_transition_vector_partial = partial(
        get_transition_vector_dcegm_two_exog_processes,
        transition_matrix=transition_matrix,
    )

    exog_savings_grid = np.linspace(0, options["max_wealth"], options["n_grid_points"])

    result_dict = solve_dcegm(
        params,
        options,
        exog_savings_grid=exog_savings_grid,
        utility_functions=utility_functions,
        budget_constraint=budget_dcegm_two_exog_processes,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
        transition_function=get_transition_vector_partial,
    )

    out = {}
    out["params"] = params
    out["options"] = options
    out["get_transition_vector_by_state"] = get_transition_vector_partial
    out["result"] = result_dict

    return out


TEST_CASES_TWO_EXOG_PROCESSES = list(
    product(list(range(WEALTH_GRID_POINTS)), list(range(8)))
)


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
    state_space, map_state_to_index = create_state_space_two_exog_processes(
        input_data_two_exog_processes["options"]
    )
    (
        state_choice_space,
        _map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat,
        _transform_between_state_and_state_choice_space,
    ) = create_state_choice_space(
        state_space,
        map_state_to_index,
        get_feasible_choice_set_two_exog_processes,
    )
    initial_conditions = {}
    state = state_space[state_idx, :]
    idxs_state_choice_combs = reshape_state_choice_vec_to_mat[state_idx]

    initial_conditions["bad_health"] = state[-1] > 1
    initial_conditions["job_offer"] = (
        state[1] == 0
    )  # working (no retirement) in period 0

    endog_grid_period = input_data_two_exog_processes["result"][state[0]]["endog_grid"]
    policy_period = input_data_two_exog_processes["result"][state[0]]["policy_left"]

    for state_choice_idx in idxs_state_choice_combs:
        choice_in_period_1 = state_choice_space[state_choice_idx][-1]

        endog_grid = endog_grid_period[state_choice_idx, wealth_idx + 1]
        policy = policy_period[state_choice_idx]

        if ~np.isnan(endog_grid) and endog_grid > 0:
            initial_conditions["wealth"] = endog_grid

            consumption = policy[wealth_idx + 1]
            diff = euler_rhs_two_exog_processes(
                initial_conditions,
                params,
                quad_draws,
                quad_weights,
                choice_in_period_1,
                consumption,
            ) - marginal_utility(consumption=consumption, rho=params["rho"])

            assert_allclose(diff, 0, atol=1e-6)
