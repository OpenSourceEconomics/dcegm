"""Model with 4 periods and the following choice structure:

- Period 3: Consume all and die (no choices).
- Period 2: Continuous consumption and work choice
  (unemployment (d=0), work (d=1), retire (d=2)).
- Period 1: Continuous consumption and work choice
  (unemployment (d=0), work (d=1)).
- Period 0: Continuous consumption and work choice
  (unemployment (d=0), work (d=1)).

Following stochastic processes:
- Health: 2 states (good (h=0), bad (h=1)); bad causes monetary health costs.
- Partner: 2 states (single (m=0), partner (m=1)); partner means higher utility of leisure.

NOTE: wage is deterministic and depends on experience and health.

State variables: period, lagged_choice, experience, health, partner.
"""

import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import get_solve_function
from dcegm.toy_models.cons_ret_model_dcegm_paper import (
    inverse_marginal_utility_crra,
    marginal_utility_crra,
    marginal_utility_final_consume_all,
    utility_crra,
)

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent


def prob_health(health, params):
    p_bad_health = (1 - health) * params[
        "p_bad_health_given_good_health"
    ] + health * params["p_bad_health_given_bad_health"]
    return jnp.array([1 - p_bad_health, p_bad_health])


def prob_partner(partner, params):
    p_marriage = (1 - partner) * (
        1 - params["p_partner_given_single"]
    ) + partner * params["p_partner_given_partner"]
    return jnp.array([1 - p_marriage, p_marriage])


def sparsity_condition(period, lagged_choice, experience, options):
    # Lagged_choice is dummy in first period
    if period == 0 and lagged_choice != 0:
        return False
    # Starting from second we check if choice was in last periods full choice set
    elif (period > 0) and lagged_choice not in choice_set(period - 1, 1):
        return False
    # Filter states with too high experience
    elif (experience > period) or (experience > options["max_experience"]):
        return False
    # If experience is 0 you can not have been working last period
    elif (experience == 0) and (lagged_choice == 1):
        return False
    # If experience is equal to period you must have been working last period (periods larger than 0)
    elif (experience == period) and (period > 0) and (lagged_choice != 1):
        return False
    else:
        return True


@pytest.fixture
def test_model():
    params = {
        # utility
        "beta": 0.95,
        "taste_shock_scale": 1,
        "sigma": 1,
        "rho": 0.5,  # CRRA coefficient
        "delta": 1,  # disutility of work
        "phi": 0.5,  # utility of joint leisure
        # budget (/ income)
        "interest_rate": 0.05,
        "constant": 1,
        "exp": 0.1,
        "exp_squared": -0.01,
        "pension_per_experience": 0.3,
        "unemployment_benefits": 0.4,
        "health_costs": 0.5,
        "consumption_floor": 0,
        # stochastic processes
        "p_bad_health_given_good_health": 0.2,
        "p_bad_health_given_bad_health": 1,
        "p_partner_given_single": 0.5,
        "p_partner_given_partner": 0.9,
    }

    options = {
        "model_params": {
            # "n_grid_points": 100,
            # "max_wealth": 500,
            "n_quad_points_stochastic": 5,
            "min_age": 0,
            "n_periods": 5,
            "n_choices": 3,
            "n_health_states": 2,
            "n_partner_states": 2,
            "max_experience": 4,
        },
        "state_space": {
            "n_periods": 5,
            "choices": np.arange(3),
            "endogenous_states": {
                "experience": np.arange(5),
            },
            "continuous_states": {
                "wealth": np.linspace(0, 500, 100),
            },
            "exogenous_processes": {
                "health": {"transition": prob_health, "states": [0, 1]},
                "partner": {"transition": prob_partner, "states": [0, 1]},
            },
        },
    }

    return params, options


def choice_set(period, lagged_choice):
    if lagged_choice == 2:
        return np.array([2])
    if period < 2:
        return np.array([0, 1])
    else:
        return np.array([0, 1, 2])


def next_period_state(period, choice, experience):
    work_ind = choice == 1
    next_state = {
        "period": period + 1,
        "lagged_choice": choice,
        "experience": experience + work_ind,
    }
    return next_state


@pytest.fixture(scope="session")
def state_space_functions():
    """Return dict with state space functions."""
    out = {
        "state_specific_choice_set": choice_set,
        "next_period_endogenous_state": next_period_state,
        "sparsity_condition": sparsity_condition,
    }
    return out


def flow_utility(consumption, choice, partner, params):
    utility_not_work = utility_crra(consumption, 1, params) + partner * params["phi"]
    utility_work = utility_crra(consumption, 0, params)
    work_ind = choice == 1
    utility = work_ind * utility_work + (1 - work_ind) * utility_not_work
    return utility


@pytest.fixture
def utility_functions():
    return {
        "utility": flow_utility,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
    }


def utility_final(wealth, params):
    return utility_crra(wealth, 1, params)


@pytest.fixture(scope="session")
def utility_functions_final_period():
    """Return dict with utility functions for final period."""
    return {
        "utility": utility_final,
        "marginal_utility": marginal_utility_final_consume_all,
    }


def budget(
    lagged_choice, experience, savings_end_of_previous_period, health, options, params
):
    unemployed = lagged_choice == 0
    working = lagged_choice == 1
    retired = lagged_choice == 2

    unemployment_income = params["unemployment_benefits"]
    retirement_income = params["pension_per_experience"] * experience
    working_income = params["constant"] * (
        1 + params["exp"] * experience + params["exp_squared"] * experience**2
    )

    income = (
        unemployed * unemployment_income
        + working * working_income
        + retired * retirement_income
        - health * params["health_costs"]
    )

    return jnp.maximum(
        income + (1 + params["interest_rate"]) * savings_end_of_previous_period,
        params["consumption_floor"],
    )


def test_extended_choice_set_model(
    test_model, state_space_functions, utility_functions, utility_functions_final_period
):
    params, options = test_model

    solve_func = get_solve_function(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        budget_constraint=budget,
        utility_functions_final_period=utility_functions_final_period,
    )
    sol = solve_func(params)
    value, _policy, _endog_grid, *_ = sol

    value_expec = pickle.load(
        open(TEST_DIR / "resources" / "extended_choice_set" / "value.pkl", "rb")
    )

    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget,
    )
    indexer = pickle.load(
        open(
            TEST_DIR
            / "resources"
            / "extended_choice_set"
            / "map_state_choice_to_index.pkl",
            "rb",
        )
    )
    state_choice_space = model["model_structure"]["state_choice_space"]
    tuple_state_choice = tuple(
        state_choice_space[:, i] for i in range(state_choice_space.shape[1])
    )
    reindex = indexer[tuple_state_choice]
    value_expec_reindexed = value_expec[reindex]

    # In the benchmark version we did not keep track if we need to augment the grid
    # in the upper envelope. Therefore, we need to loop over state choices and filter
    # the arrays
    for i in range(value.shape[0]):
        # We can't use the last period
        if state_choice_space[i, 0] < 4:
            # Read out relevant row of arrays and compare first two elements
            value_i = value[i]
            value_expec_i = value_expec_reindexed[i]
            assert_allclose(value_i[:2], value_expec_i[:2])
            # Now check all elements that are not nan in the arrays and do not equal the
            # second element in the expected array are equal
            value_i_non_nan = value_i[~np.isnan(value_i)][2:]
            value_expec_i_non_nan = value_expec_i[~np.isnan(value_expec_i)][2:]
            value_expec_i_non_nan = value_expec_i_non_nan[
                ~np.isclose(value_expec_i_non_nan, value_expec_i[1])
            ]
            assert_allclose(value_i_non_nan, value_expec_i_non_nan)
