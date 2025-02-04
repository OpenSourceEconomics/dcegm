from itertools import product
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.final_periods import solve_final_period
from dcegm.law_of_motion import calc_cont_grids_next_period
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import create_solution_container, solve_dcegm
from dcegm.solve_single_period import solve_for_interpolated_values
from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
    utility_crra,
)

N_PERIODS = 2
MAX_WEALTH = 50
WEALTH_GRID_POINTS = 100
EXPERIENCE_GRID_POINTS = 6

ALL_WEALTH_GRIDS = list(range(WEALTH_GRID_POINTS))
RANDOM_TEST_SET = np.random.choice(ALL_WEALTH_GRIDS, size=10, replace=False)
PRODUCT = list(product(RANDOM_TEST_SET, list(range(2))))
WEALTH_AND_STATE_IDX = [tup for tup in PRODUCT]

PARAMS = {
    "beta": 0.95,
    "delta": 0.35,
    "rho": 1.95,
    "interest_rate": 0.04,
    "lambda": 1,  # taste shock (scale) parameter
    "sigma": 1,  # shock on labor income, standard deviation
    "constant": 0.75,
    "exp": 0.04,
    "exp_squared": -0.0002,
    "consumption_floor": 0.001,
}


# ====================================================================================
# Model functions
# ====================================================================================


def marginal_utility_crra(
    consumption: jnp.array, params: Dict[str, float]
) -> jnp.array:
    """Computes marginal utility of CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        marginal_utility (jnp.array): Marginal utility of CRRA consumption
            function. Array of shape (n_quad_stochastic * n_grid_wealth,).

    """
    marginal_utility = consumption ** (-params["rho"])

    return marginal_utility


def euler_rhs(
    lagged_wealth,
    lagged_consumption,
    lagged_choice,
    experience,
    income_shocks,
    weights,
    params,
):
    beta = params["beta"]
    interest_factor = 1 + params["interest_rate"]

    rhs = 0
    for idx_draw, draw in enumerate(income_shocks):
        marg_util_draw = marginal_utility_weighted(
            lagged_wealth=lagged_wealth,
            params=params,
            lagged_choice=lagged_choice,
            income_shock=draw,
            lagged_consumption=lagged_consumption,
            experience=experience,
        )
        rhs += weights[idx_draw] * marg_util_draw

    return rhs * beta * interest_factor


def marginal_utility_weighted(
    lagged_wealth,
    lagged_choice,
    lagged_consumption,
    experience,
    income_shock,
    params,
):
    """Return the expected marginal utility for one realization of the wage shock."""
    exp_new = next_period_experience(
        period=1, lagged_choice=lagged_choice, experience=experience, params=params
    )

    budget_next = budget_constraint_continuous(
        period=1,
        lagged_wealth=lagged_wealth,
        lagged_consumption=lagged_consumption,
        lagged_choice=lagged_choice,
        experience=exp_new,
        income_shock_previous_period=income_shock,
        options={},
        params=params,
    )

    marginal_utility_weighted = 0
    for choice_next in (0, 1):
        marginal_utility = marginal_utility_crra(consumption=budget_next, params=params)

        marginal_utility_weighted += (
            choice_prob(consumption=budget_next, choice=choice_next, params=params)
            * marginal_utility
        )

    return marginal_utility_weighted


def choice_prob(consumption, choice, params):
    v = utility_crra(consumption=consumption, params=params, choice=choice)
    v_other = utility_crra(consumption=consumption, params=params, choice=1 - choice)
    max_v = jnp.maximum(v, v_other)

    return np.exp((v - max_v) / params["lambda"]) / (
        np.exp((v_other - max_v) / params["lambda"])
        + np.exp((v - max_v) / params["lambda"])
    )


def budget_constraint_continuous(
    period: int,
    lagged_wealth: float,
    lagged_consumption: float,
    lagged_choice: int,
    experience: float,
    income_shock_previous_period: float,
    options: Dict[str, Any],
    params: Dict[str, float],
) -> float:

    working = lagged_choice == 0

    experience_years = experience * period

    income_from_previous_period = calc_stochastic_income(
        experience=experience_years,
        wage_shock=income_shock_previous_period,
        params=params,
    )

    wealth_beginning_of_period = income_from_previous_period * working + (
        1 + params["interest_rate"]
    ) * (lagged_wealth - lagged_consumption)

    # Retirement safety net, only in retirement model, but we require to have it always
    # as a parameter
    return jnp.maximum(wealth_beginning_of_period, params["consumption_floor"])


def budget_constraint_continuous_dcegm(
    period: int,
    savings_end_of_previous_period: float,
    lagged_choice: int,
    experience: float,
    income_shock_previous_period: float,
    options: Dict[str, Any],
    params: Dict[str, float],
) -> float:

    working = lagged_choice == 0

    experience_years = experience * period

    income_from_previous_period = calc_stochastic_income(
        experience=experience_years,
        wage_shock=income_shock_previous_period,
        params=params,
    )

    wealth_beginning_of_period = (
        income_from_previous_period * working
        + (1 + params["interest_rate"]) * savings_end_of_previous_period
    )

    # Retirement safety net, only in retirement model, but we require to have it always
    # as a parameter
    return jnp.maximum(wealth_beginning_of_period, params["consumption_floor"])


def calc_stochastic_income(
    experience: int,
    wage_shock: float,
    params: Dict[str, float],
) -> float:

    labor_income = (
        params["constant"]
        + params["exp"] * experience
        + params["exp_squared"] * experience**2
    )

    return jnp.exp(labor_income + wage_shock)


def next_period_experience(period, lagged_choice, experience, params):
    return (1 / period) * ((period - 1) * experience + (lagged_choice == 0))


# ====================================================================================
# Test inputs
# ====================================================================================


@pytest.fixture(scope="module")
def create_test_inputs():
    options = {}
    _raw_options = {
        "n_discrete_choices": 2,
        "quadrature_points_stochastic": 5,
    }
    params = PARAMS

    options["model_params"] = _raw_options
    options["model_params"]["n_periods"] = N_PERIODS
    options["model_params"]["max_wealth"] = MAX_WEALTH
    options["model_params"]["n_grid_points"] = WEALTH_GRID_POINTS
    options["model_params"]["n_choices"] = _raw_options["n_discrete_choices"]

    options["state_space"] = {
        "n_periods": N_PERIODS,
        "choices": np.arange(2),
        "continuous_states": {
            "wealth": jnp.linspace(
                0,
                MAX_WEALTH,
                WEALTH_GRID_POINTS,
            ),
            "experience": jnp.linspace(0, 1, EXPERIENCE_GRID_POINTS),
        },
    }

    utility_functions = create_utility_function_dict()
    utility_functions_final_period = create_final_period_utility_function_dict()

    # =================================================================================
    # Continuous experience
    # =================================================================================

    state_space_functions = {
        "next_period_experience": next_period_experience,
    }

    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_continuous_dcegm,
    )

    (
        cont_grids_next_period,
        income_shock_draws_unscaled,
        income_shock_weights,
        taste_shock_scale,
        exog_grids_cont,
        model_funcs_cont,
        last_two_period_batch_info_cont,
        value_solved,
        policy_solved,
        endog_grid_solved,
    ) = _get_solve_last_two_periods_args(
        model, params, has_second_continuous_state=True
    )

    (
        value_solved,
        policy_solved,
        endog_grid_solved,
        value_interp_final_period,
        marginal_utility_final_last_period,
    ) = solve_final_period(
        idx_state_choices_final_period=last_two_period_batch_info_cont[
            "idx_state_choices_final_period"
        ],
        idx_parent_states_final_period=last_two_period_batch_info_cont[
            "idxs_parent_states_final_period"
        ],
        state_choice_mat_final_period=last_two_period_batch_info_cont[
            "state_choice_mat_final_period"
        ],
        cont_grids_next_period=cont_grids_next_period,
        exog_grids=exog_grids_cont,
        params=params,
        model_funcs=model_funcs_cont,
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        has_second_continuous_state=True,
    )

    endog_grid, policy, value_second_last = solve_for_interpolated_values(
        value_interpolated=value_interp_final_period,
        marginal_utility_interpolated=marginal_utility_final_last_period,
        state_choice_mat=last_two_period_batch_info_cont[
            "state_choice_mat_second_last_period"
        ],
        child_state_idxs=last_two_period_batch_info_cont[
            "child_states_second_last_period"
        ],
        states_to_choices_child_states=last_two_period_batch_info_cont[
            "state_to_choices_final_period"
        ],
        params=params,
        taste_shock_scale=jnp.array([taste_shock_scale]),
        income_shock_weights=income_shock_weights,
        exog_grids=exog_grids_cont,
        model_funcs=model_funcs_cont,
        has_second_continuous_state=True,
    )

    idx_second_last = last_two_period_batch_info_cont[
        "idx_state_choices_second_last_period"
    ]

    value_solved = value_solved.at[idx_second_last, ...].set(value_second_last)
    policy_solved = policy_solved.at[idx_second_last, ...].set(policy)
    endog_grid_solved = endog_grid_solved.at[idx_second_last, ...].set(endog_grid)

    return (
        value_solved,
        policy_solved,
        endog_grid_solved,
        model,
        params,
        income_shock_draws_unscaled,
        income_shock_weights,
        utility_functions,
        utility_functions_final_period,
        state_space_functions,
    )


# ====================================================================================
# Tests
# ====================================================================================


def test_solution(create_test_inputs):

    (
        value_solved,
        policy_solved,
        endog_grid_solved,
        model,
        params,
        _income_shock_draws_unscaled,
        _income_shock_weights,
        utility_functions,
        utility_functions_final_period,
        state_space_functions,
    ) = create_test_inputs

    value_dcegm, policy_dcegm, endog_grid_dcegm = solve_dcegm(
        params,
        model["options"],
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_continuous_dcegm,
    )

    aaae(value_dcegm, value_solved)
    aaae(policy_dcegm, policy_solved)
    aaae(endog_grid_dcegm, endog_grid_solved)


@pytest.mark.parametrize("wealth_idx, state_idx", WEALTH_AND_STATE_IDX)
def test_euler_equation(wealth_idx, state_idx, create_test_inputs):

    (
        _value_solved,
        policy_solved,
        endog_grid_solved,
        model,
        params,
        income_shock_draws_unscaled,
        income_shock_weights,
        *_,
    ) = create_test_inputs

    model_structure = model["model_structure"]
    state_choice_space = model["model_structure"]["state_choice_space"]
    state_choice_space_period_0 = state_choice_space[state_choice_space[:, 0] == 0]

    parent_states_of_current_state = np.where(
        model_structure["map_state_choice_to_parent_state"] == state_idx
    )[0]

    for state_choice_idx in parent_states_of_current_state:
        for exp_idx, exp in enumerate(range(EXPERIENCE_GRID_POINTS)):
            endog_grid_period_0 = endog_grid_solved[
                state_choice_idx, exp_idx, wealth_idx + 1
            ]
            policy_period_0 = policy_solved[state_choice_idx, exp_idx, wealth_idx + 1]
            lagged_choice = state_choice_space_period_0[state_choice_idx, -1]

            if ~np.isnan(endog_grid_period_0) and endog_grid_period_0 > 0:

                euler_next = euler_rhs(
                    lagged_wealth=endog_grid_period_0,
                    lagged_consumption=policy_period_0,
                    lagged_choice=lagged_choice,
                    experience=exp,
                    income_shocks=income_shock_draws_unscaled * params["sigma"],
                    weights=income_shock_weights,
                    params=params,
                )

                marg_util_current = marginal_utility_crra(
                    consumption=policy_period_0, params=params
                )

                assert_allclose(euler_next - marg_util_current, 0, atol=1e-6)


# ====================================================================================
# Auxiliary functions
# ====================================================================================


def _get_solve_last_two_periods_args(model, params, has_second_continuous_state):
    options = model["options"]
    batch_info_last_two_periods = model["batch_info"]["last_two_period_info"]

    exog_grids = options["exog_grids"]

    # Prepare income shock draws and scaling
    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        options["model_params"]["quadrature_points_stochastic"]
    )
    taste_shock_scale = params["lambda"]

    # Get state space dictionary and model functions
    state_space_dict = model["model_structure"]["state_space_dict"]
    model_funcs = model["model_funcs"]

    cont_grids_next_period = calc_cont_grids_next_period(
        state_space_dict=state_space_dict,
        exog_grids=exog_grids,
        income_shock_draws_unscaled=income_shock_draws_unscaled,
        params=params,
        model_funcs=model_funcs,
        has_second_continuous_state=has_second_continuous_state,
    )

    # Create solution containers for value, policy, and endogenous grids
    value_solved, policy_solved, endog_grid_solved = create_solution_container(
        n_state_choices=model["model_structure"]["state_choice_space"].shape[0],
        options=options,
        has_second_continuous_state=has_second_continuous_state,
    )

    return (
        cont_grids_next_period,
        income_shock_draws_unscaled,
        income_shock_weights,
        taste_shock_scale,
        exog_grids,
        model_funcs,
        batch_info_last_two_periods,
        value_solved,
        policy_solved,
        endog_grid_solved,
    )
