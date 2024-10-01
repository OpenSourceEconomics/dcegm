from itertools import product
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.final_periods import solve_final_period, solve_last_two_periods
from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from dcegm.law_of_motion import (
    calculate_continuous_state,
    calculate_resources,
    calculate_resources_for_second_continuous_state,
)
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import create_solution_container, solve_dcegm
from dcegm.solve_single_period import solve_for_interpolated_values
from tests.utils.interp1d_auxiliary import (
    interpolate_policy_and_value_on_wealth_grid,
    linear_interpolation_with_extrapolation,
)
from toy_models.consumption_retirement_model.utility_functions import (
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
    lagged_resources,
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
            lagged_resources=lagged_resources,
            params=params,
            lagged_choice=lagged_choice,
            income_shock=draw,
            lagged_consumption=lagged_consumption,
            experience=experience,
        )
        rhs += weights[idx_draw] * marg_util_draw

    return rhs * beta * interest_factor


def marginal_utility_weighted(
    lagged_resources,
    lagged_choice,
    lagged_consumption,
    experience,
    income_shock,
    params,
):
    """Return the expected marginal utility for one realization of the wage shock."""
    exp_new = get_next_period_experience(
        period=1, lagged_choice=lagged_choice, experience=experience, params=params
    )

    budget_next = budget_constraint_continuous(
        period=1,
        lagged_resources=lagged_resources,
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
    lagged_resources: float,
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
    ) * (lagged_resources - lagged_consumption)

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


def get_next_period_experience(period, lagged_choice, experience, params):
    # ToDo: Rewrite in the sense of budget equation

    return (1 / period) * ((period - 1) * experience + (lagged_choice == 0))


def get_next_period_discrete_state(period, choice):

    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    return next_state


def get_state_specific_feasible_choice_set(
    lagged_choice: int,
    options: Dict,
) -> np.ndarray:
    """Select state-specific feasible choice set such that retirement is absorbing."""

    n_choices = options["n_choices"]

    # # Once the agent choses retirement, she can only choose retirement thereafter.
    # # Hence, retirement is an absorbing state.
    # if lagged_choice == 1:
    #     feasible_choice_set = np.array([1])
    # else:
    feasible_choice_set = np.arange(n_choices)

    return feasible_choice_set


# ====================================================================================
# Test inputs
# ====================================================================================


@pytest.fixture()
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
        "get_next_period_state": get_next_period_discrete_state,
        "update_continuous_state": get_next_period_experience,
        "get_state_specific_feasible_choice_set": get_state_specific_feasible_choice_set,
    }

    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_continuous_dcegm,
    )

    (
        wealth_and_continuous_state_next_period_cont,
        income_shock_draws_unscaled,
        income_shock_weights,
        taste_shock_scale,
        exog_grids_cont,
        wealth_beginning_at_regular_cont,
        model_funcs_cont,
        batch_info_cont,
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
        _,
    ) = solve_final_period(
        idx_state_choices_final_period=batch_info_cont[
            "idx_state_choices_final_period"
        ],
        idx_parent_states_final_period=batch_info_cont[
            "idxs_parent_states_final_period"
        ],
        state_choice_mat_final_period=batch_info_cont["state_choice_mat_final_period"],
        wealth_and_continuous_state_next_period=wealth_and_continuous_state_next_period_cont,
        wealth_beginning_at_regular_period=wealth_beginning_at_regular_cont,
        params=params,
        compute_utility=model_funcs_cont["compute_utility_final"],
        compute_marginal_utility=model_funcs_cont["compute_marginal_utility_final"],
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        has_second_continuous_state=True,
    )

    endog_grid, policy, value_second_last, *_ = solve_for_interpolated_values(
        value_interpolated=value_interp_final_period,
        marginal_utility_interpolated=marginal_utility_final_last_period,
        state_choice_mat=batch_info_cont["state_choice_mat_second_last_period"],
        child_state_idxs=batch_info_cont["child_states_second_last_period"],
        states_to_choices_child_states=batch_info_cont["state_to_choices_final_period"],
        params=params,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
        exog_grids=exog_grids_cont["wealth"],
        model_funcs=model_funcs_cont,
        has_second_continuous_state=True,
    )

    idx_second_last = batch_info_cont["idx_state_choices_second_last_period"]

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
                    lagged_resources=endog_grid_period_0,
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
    batch_info = model["batch_info"]

    exog_grids = options["exog_grids"]

    # Prepare income shock draws and scaling
    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        options["model_params"]["quadrature_points_stochastic"]
    )
    taste_shock_scale = params["lambda"]

    # Get state space dictionary and model functions
    state_space_dict = model["model_structure"]["state_space_dict"]
    model_funcs = model["model_funcs"]

    # Distinction based on has_second_continuous_state
    if has_second_continuous_state:
        # Calculate continuous state for the next period
        continuous_state_next_period = calculate_continuous_state(
            discrete_states_beginning_of_period=state_space_dict,
            continuous_grid=exog_grids["second_continuous"],
            params=params,
            compute_continuous_state=model_funcs[
                "compute_beginning_of_period_continuous_state"
            ],
        )

        # Extra dimension for continuous state
        wealth_beginning_of_next_period = (
            calculate_resources_for_second_continuous_state(
                discrete_states_beginning_of_next_period=state_space_dict,
                continuous_state_beginning_of_next_period=continuous_state_next_period,
                savings_grid=exog_grids["wealth"],
                income_shocks=income_shock_draws_unscaled * params["sigma"],
                params=params,
                compute_beginning_of_period_resources=model_funcs[
                    "compute_beginning_of_period_resources"
                ],
            )
        )

        # Extra dimension for continuous state (regular wealth calculation)
        wealth_beginning_at_regular = calculate_resources_for_second_continuous_state(
            discrete_states_beginning_of_next_period=state_space_dict,
            continuous_state_beginning_of_next_period=jnp.tile(
                exog_grids["second_continuous"],
                (continuous_state_next_period.shape[0], 1),
            ),
            savings_grid=exog_grids["wealth"],
            income_shocks=income_shock_draws_unscaled * params["sigma"],
            params=params,
            compute_beginning_of_period_resources=model_funcs[
                "compute_beginning_of_period_resources"
            ],
        )

        # Combined wealth and continuous state for the next period
        wealth_and_continuous_state_next_period = (
            continuous_state_next_period,
            wealth_beginning_of_next_period,
        )

    else:
        # Single state calculation (no second continuous state)
        wealth_and_continuous_state_next_period = calculate_resources(
            discrete_states_beginning_of_period=state_space_dict,
            savings_grid=exog_grids["wealth"],
            income_shocks_current_period=income_shock_draws_unscaled * params["sigma"],
            params=params,
            compute_beginning_of_period_resources=model_funcs[
                "compute_beginning_of_period_resources"
            ],
        )
        wealth_beginning_at_regular = None

    # Create solution containers for value, policy, and endogenous grids
    value_solved, policy_solved, endog_grid_solved = create_solution_container(
        n_state_choices=model["model_structure"]["state_choice_space"].shape[0],
        options=options,
        has_second_continuous_state=has_second_continuous_state,
    )

    return (
        wealth_and_continuous_state_next_period,
        income_shock_draws_unscaled,
        income_shock_weights,
        taste_shock_scale,
        exog_grids,
        wealth_beginning_at_regular,
        model_funcs,
        batch_info,
        value_solved,
        policy_solved,
        endog_grid_solved,
    )