from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import solve_dcegm
from tests.utils.interp1d_auxiliary import (
    interpolate_policy_and_value_on_wealth_grid,
    linear_interpolation_with_extrapolation,
)
from toy_models.consumption_retirement_model.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)

N_PERIODS = 20
MAX_WEALTH = 50
WEALTH_GRID_POINTS = 100
EXPERIENCE_GRID_POINTS = 6


PARAMS = {
    "beta": 0.95,
    "delta": 0.35,
    "rho": 1.95,
    "savings_rate": 0.04,
    "constant": 0.75,
    "exp": 0.04,
    "exp_squared": -0.0002,
    "consumption_floor": 0.001,
}


# ====================================================================================
# Model functions
# ====================================================================================


def budget_constraint_continuous(
    period: int,
    lagged_choice: int,
    experience: float,
    savings_end_of_previous_period: float,
    income_shock_previous_period: float,
    options: Dict[str, Any],
    params: Dict[str, float],
) -> float:

    working = lagged_choice == 0

    experience_years = experience * period

    income_from_previous_period = _calc_stochastic_income(
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


def budget_constraint_discrete(
    lagged_choice: int,
    experience: int,
    savings_end_of_previous_period: float,
    income_shock_previous_period: float,
    options: Dict[str, Any],
    params: Dict[str, float],
) -> float:

    working = lagged_choice == 0

    income_from_previous_period = _calc_stochastic_income(
        experience=experience,
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


@jax.jit
def _calc_stochastic_income(
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


def get_next_period_experience(period, choice, experience, options, params):

    working = choice == 0

    return 1 / (period + 1) * (period * experience + working)


def get_next_period_state(period, choice, married, experience):

    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice
    next_state["married"] = married

    next_state["experience"] = experience + (choice == 0)

    return next_state


def get_next_period_discrete_state(period, choice, married):

    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice
    next_state["married"] = married

    return next_state


def get_state_specific_feasible_choice_set(
    lagged_choice: int,
    options: Dict,
) -> np.ndarray:
    """Select state-specific feasible choice set such that retirement is absorbing."""

    n_choices = options["n_choices"]

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if lagged_choice == 1:
        feasible_choice_set = np.array([1])
    else:
        feasible_choice_set = np.arange(n_choices)

    return feasible_choice_set


def sparsity_condition(
    period,
    experience,
    options,
):

    max_init_experience = 0

    cond = True

    if (period + max_init_experience < experience) | (
        experience > options["n_periods"]
    ):
        cond = False

    return cond


# ====================================================================================
# Test
# ====================================================================================


@pytest.mark.xfail(reason="Find bug")
def test_replication_discrete_versus_continuous_experience(load_example_model):
    options = {}
    model_name = "retirement_no_taste_shocks"
    params, _raw_options = load_example_model(f"{model_name}")

    options["model_params"] = _raw_options
    options["model_params"]["n_periods"] = N_PERIODS
    options["model_params"]["max_wealth"] = MAX_WEALTH
    options["model_params"]["n_grid_points"] = WEALTH_GRID_POINTS
    options["model_params"]["n_choices"] = _raw_options["n_discrete_choices"]

    options["state_space"] = {
        "n_periods": N_PERIODS,
        "choices": np.arange(2),
        "endogenous_states": {
            "experience": np.arange(N_PERIODS),
            "married": np.arange(2),
            "sparsity_condition": sparsity_condition,
        },
        "continuous_states": {
            "wealth": jnp.linspace(
                0,
                MAX_WEALTH,
                WEALTH_GRID_POINTS,
            )
        },
    }

    utility_functions = create_utility_function_dict()
    utility_functions_final_period = create_final_period_utility_function_dict()

    state_space_functions = {
        "get_next_period_state": get_next_period_state,
        "get_state_specific_feasible_choice_set": get_state_specific_feasible_choice_set,
    }

    model_disc = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_discrete,
    )
    value_disc, policy_disc, endog_grid_disc = solve_dcegm(
        params,
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_discrete,
    )

    state_choice_space_disc = model_disc["model_structure"]["state_choice_space"]
    where_experience = model_disc["model_structure"]["state_space_names"].index(
        "experience"
    )

    # =================================================================================
    # Continuous experience
    # =================================================================================
    experience_grid = jnp.linspace(0, 1, EXPERIENCE_GRID_POINTS)

    options_cont = options.copy()
    options_cont["state_space"]["continuous_states"]["experience"] = experience_grid
    options_cont["state_space"]["endogenous_states"].pop("experience")
    options_cont["state_space"]["endogenous_states"].pop("sparsity_condition")

    state_space_functions_continuous = {
        "get_next_period_state": get_next_period_discrete_state,
        "update_continuous_state": get_next_period_experience,
        "get_state_specific_feasible_choice_set": get_state_specific_feasible_choice_set,
    }

    model_cont = setup_model(
        options=options_cont,
        state_space_functions=state_space_functions_continuous,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_continuous,
    )
    value_cont, policy_cont, endog_grid_cont = solve_dcegm(
        params,
        options_cont,
        state_space_functions=state_space_functions_continuous,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_continuous,
    )

    state_choice_space_cont = model_cont["model_structure"]["state_choice_space"]

    # =================================================================================
    # Interpolate
    # =================================================================================

    period = 15
    experience = 10
    exp_share_to_test = experience / period

    state_choice_disc = state_choice_space_disc[
        (state_choice_space_disc[:, 0] == period)
        & (state_choice_space_disc[:, where_experience] == experience)
    ][-1]
    state_choice_cont = state_choice_space_cont[
        state_choice_space_cont[:, 0] == period
    ][-1]

    idx_disc = jnp.where(jnp.all(state_choice_space_disc == state_choice_disc, axis=1))[
        0
    ]
    idx_cont = jnp.where(jnp.all(state_choice_space_cont == state_choice_cont, axis=1))[
        0
    ]

    state_space_names_disc = model_disc["model_structure"]["state_space_names"]
    state_space_names_disc.append("choice")
    state_choice_vec_disc = dict(zip(state_space_names_disc, state_choice_disc))

    state_space_names_cont = model_cont["model_structure"]["state_space_names"]
    state_space_names_cont.append("choice")
    state_choice_vec_cont = dict(zip(state_space_names_cont, state_choice_cont))

    for exp in range(1, 6):
        for wealth in range(1, options["model_params"]["n_grid_points"]):

            policy_cont_interp, value_cont_interp = (
                interp2d_policy_and_value_on_wealth_and_regular_grid(
                    regular_grid=experience_grid,
                    wealth_grid=jnp.squeeze(endog_grid_cont[idx_cont], axis=0),
                    policy_grid=jnp.squeeze(policy_cont[idx_cont], axis=0),
                    value_grid=jnp.squeeze(value_cont[idx_cont], axis=0),
                    regular_point_to_interp=exp_share_to_test,
                    wealth_point_to_interp=jnp.squeeze(
                        endog_grid_cont[idx_cont, exp, wealth], axis=0
                    ),
                    compute_utility=model_cont["model_funcs"]["compute_utility"],
                    state_choice_vec=state_choice_vec_cont,
                    params=params,
                )
            )

            # (
            #     policy_disc_interp,
            #     value_disc_interp,
            # ) = interpolate_policy_and_value_on_wealth_grid(
            #     wealth_beginning_of_period=jnp.squeeze(
            #         endog_grid_cont[idx_disc, exp, wealth], axis=0
            #     ),
            #     endog_wealth_grid=jnp.squeeze(endog_grid_disc[idx_disc], axis=0),
            #     policy=jnp.squeeze(policy_disc[idx_disc], axis=0),
            #     value=jnp.squeeze(value_disc[idx_disc], axis=0),
            # )

            policy_disc_interp, value_disc_interp = interp1d_policy_and_value_on_wealth(
                wealth=jnp.squeeze(endog_grid_cont[idx_disc, exp, wealth], axis=0),
                endog_grid=jnp.squeeze(endog_grid_disc[idx_disc], axis=0),
                policy=jnp.squeeze(policy_disc[idx_disc], axis=0),
                value=jnp.squeeze(value_disc[idx_disc], axis=0),
                compute_utility=model_disc["model_funcs"]["compute_utility"],
                state_choice_vec=state_choice_vec_disc,
                params=params,
            )

            aaae(value_cont_interp, value_disc_interp)
            aaae(policy_cont_interp, policy_disc_interp)
