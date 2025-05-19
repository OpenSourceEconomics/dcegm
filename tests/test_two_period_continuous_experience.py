from itertools import product
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
from dcegm.backward_induction import create_solution_container
from dcegm.final_periods import solve_final_period
from dcegm.law_of_motion import calc_cont_grids_next_period
from dcegm.numerical_integration import quadrature_legendre
from dcegm.solve_single_period import solve_for_interpolated_values

MAX_WEALTH = 50
WEALTH_GRID_POINTS = 100
EXPERIENCE_GRID_POINTS = 6

ALL_WEALTH_GRIDS = list(range(WEALTH_GRID_POINTS))
RANDOM_TEST_SET = np.random.choice(ALL_WEALTH_GRIDS, size=10, replace=False)
PRODUCT = list(product(RANDOM_TEST_SET, list(range(2))))
WEALTH_AND_STATE_IDX = [tup for tup in PRODUCT]


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
        params=params,
    )
    model_functions = toy_models.load_example_model_functions("dcegm_paper")
    utility_functions = model_functions["utility_functions"]

    marginal_utility_weighted = 0
    for choice_next in (0, 1):
        marginal_utility = utility_functions["marginal_utility"](
            consumption=budget_next, params=params
        )

        marginal_utility_weighted += (
            choice_prob(
                consumption=budget_next,
                choice=choice_next,
                params=params,
                utility_function=utility_functions["utility"],
            )
            * marginal_utility
        )

    return marginal_utility_weighted


def choice_prob(consumption, choice, params, utility_function):

    v = utility_function(consumption=consumption, params=params, choice=choice)
    v_other = utility_function(
        consumption=consumption, params=params, choice=1 - choice
    )
    max_v = jnp.maximum(v, v_other)

    return np.exp((v - max_v) / params["taste_shock_scale"]) / (
        np.exp((v_other - max_v) / params["taste_shock_scale"])
        + np.exp((v - max_v) / params["taste_shock_scale"])
    )


def budget_constraint_continuous(
    period: int,
    lagged_wealth: float,
    lagged_consumption: float,
    lagged_choice: int,
    experience: float,
    income_shock_previous_period: float,
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
    asset_end_of_previous_period: float,
    lagged_choice: int,
    experience: float,
    income_shock_previous_period: float,
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
        + (1 + params["interest_rate"]) * asset_end_of_previous_period
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
    params = {
        "beta": 0.95,
        "delta": 0.35,
        "rho": 1.95,
        "interest_rate": 0.04,
        "taste_shock_scale": 1,  # taste shock (scale) parameter
        "sigma": 1,  # shock on labor income, standard deviation
        "income_shock_mean": 0,  # shock on labor income, mean
        "constant": 0.75,
        "exp": 0.04,
        "exp_squared": -0.0002,
        "consumption_floor": 0.001,
    }

    model_specs = {
        "n_periods": 2,
        "n_discrete_choices": 2,
    }

    model_config = {
        "n_periods": 2,
        "choices": np.arange(2),
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(
                0,
                50,
                100,
            ),
            "experience": jnp.linspace(0, 1, EXPERIENCE_GRID_POINTS),
        },
        "n_quad_points": 5,
    }

    model_functions = toy_models.load_example_model_functions("dcegm_paper")
    utility_functions = model_functions["utility_functions"]

    utility_functions_final_period = model_functions["utility_functions_final_period"]

    # =================================================================================
    # Continuous experience
    # =================================================================================

    state_space_functions = {
        "next_period_experience": next_period_experience,
    }

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_continuous_dcegm,
    )
    model_config = model.model_config

    (
        cont_grids_next_period,
        income_shock_draws_unscaled,
        income_shock_weights,
        taste_shock_scale,
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
        continuous_states_info=model_config["continuous_states_info"],
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
        taste_shock_scale_is_scalar=True,
        income_shock_weights=income_shock_weights,
        continuous_grids_info=model_config["continuous_states_info"],
        model_funcs=model_funcs_cont,
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

    model_solved = model.solve(params)

    aaae(model_solved.value, value_solved)
    aaae(model_solved.policy, policy_solved)
    aaae(model_solved.endog_grid, endog_grid_solved)


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

    model_structure = model.model_structure
    state_choice_space = model_structure["state_choice_space"]
    state_choice_space_period_0 = state_choice_space[state_choice_space[:, 0] == 0]

    parent_states_of_current_state = np.where(
        model_structure["map_state_choice_to_parent_state"] == state_idx
    )[0]

    model_functions = toy_models.load_example_model_functions("dcegm_paper")
    utility_functions = model_functions["utility_functions"]

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

                marg_util_current = utility_functions["marginal_utility"](
                    consumption=policy_period_0, params=params
                )

                assert_allclose(euler_next - marg_util_current, 0, atol=1e-6)


# ====================================================================================
# Auxiliary functions
# ====================================================================================


def _get_solve_last_two_periods_args(model, params, has_second_continuous_state):
    model_config = model.model_config
    batch_info = model.batch_info
    batch_info_last_two_periods = batch_info["last_two_period_info"]

    # Prepare income shock draws and scaling
    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        model_config["n_quad_points"]
    )
    taste_shock_scale = params["taste_shock_scale"]

    # Get state space dictionary and model functions
    model_structure = model.model_structure
    state_space_dict = model_structure["state_space_dict"]
    model_funcs = model.model_funcs

    cont_grids_next_period = calc_cont_grids_next_period(
        state_space_dict=state_space_dict,
        model_config=model_config,
        income_shock_draws_unscaled=income_shock_draws_unscaled,
        params=params,
        model_funcs=model_funcs,
        has_second_continuous_state=has_second_continuous_state,
    )

    n_total_wealth_grid = model_config["tuning_params"]["n_total_wealth_grid"]

    if has_second_continuous_state:
        n_second_continuous_grid = model_config["continuous_states_info"][
            "n_second_continuous_grid"
        ]
    else:
        n_second_continuous_grid = None

    # Create solution containers for value, policy, and endogenous grids
    value_solved, policy_solved, endog_grid_solved = create_solution_container(
        n_state_choices=model_structure["state_choice_space"].shape[0],
        n_total_wealth_grid=n_total_wealth_grid,
        n_second_continuous_grid=n_second_continuous_grid,
        has_second_continuous_state=has_second_continuous_state,
    )

    return (
        cont_grids_next_period,
        income_shock_draws_unscaled,
        income_shock_weights,
        taste_shock_scale,
        model_funcs,
        batch_info_last_two_periods,
        value_solved,
        policy_solved,
        endog_grid_solved,
    )
