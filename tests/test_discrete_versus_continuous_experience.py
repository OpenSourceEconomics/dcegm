from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import get_solve_func_for_model
from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)
from toy_models.cons_ret_model_with_exp.state_space_objects import (
    create_state_space_function_dict,
)

N_PERIODS = 5
N_DISCRETE_CHOICES = 2
MAX_WEALTH = 50
WEALTH_GRID_POINTS = 100
EXPERIENCE_GRID_POINTS = 6


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


def budget_constraint_continuous(
    period,
    lagged_choice,
    experience,
    savings_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    experience_years = experience * period
    return budget_constraint_discrete(
        lagged_choice=lagged_choice,
        experience=experience_years,
        savings_end_of_previous_period=savings_end_of_previous_period,
        income_shock_previous_period=income_shock_previous_period,
        params=params,
    )


def budget_constraint_discrete(
    lagged_choice,
    experience,
    savings_end_of_previous_period,
    income_shock_previous_period,
    params,
):

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


def _calc_stochastic_income(
    experience,
    wage_shock,
    params,
):

    labor_income = (
        params["constant"]
        + params["exp"] * experience
        + params["exp_squared"] * experience**2
    )

    return jnp.exp(labor_income + wage_shock)


def get_next_period_experience(period, lagged_choice, experience):
    return (1 / period) * ((period - 1) * experience + (lagged_choice == 0))


def sparsity_condition(
    period,
    experience,
    options,
):

    max_init_experience = 0

    cond = True

    if (period + max_init_experience < experience) | (
        experience >= options["n_periods"]
    ):
        cond = False

    return cond


# ====================================================================================
# Test
# ====================================================================================


@pytest.fixture(scope="session")
def test_setup():
    options = {}
    _raw_options = {
        "n_discrete_choices": N_DISCRETE_CHOICES,
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
        "choices": np.arange(
            N_DISCRETE_CHOICES,
        ),
        "endogenous_states": {
            "experience": np.arange(N_PERIODS),
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

    # =================================================================================
    # Discrete experience
    # =================================================================================

    state_space_functions_discrete = create_state_space_function_dict()
    model_disc = setup_model(
        options=options,
        state_space_functions=state_space_functions_discrete,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_discrete,
    )

    solve_disc = get_solve_func_for_model(model_disc)
    value_disc, policy_disc, endog_grid_disc = solve_disc(params)

    # =================================================================================
    # Continuous experience
    # =================================================================================

    experience_grid = jnp.linspace(0, 1, EXPERIENCE_GRID_POINTS)

    options_cont = options.copy()
    options_cont["state_space"]["continuous_states"]["experience"] = experience_grid
    options_cont["state_space"]["endogenous_states"].pop("experience")
    options_cont["state_space"]["endogenous_states"].pop("sparsity_condition")

    state_space_functions_continuous = {
        "update_continuous_state": get_next_period_experience,
    }

    model_cont = setup_model(
        options=options_cont,
        state_space_functions=state_space_functions_continuous,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_continuous,
    )

    solve_cont = get_solve_func_for_model(model_cont)
    value_cont, policy_cont, endog_grid_cont = solve_cont(params)

    return (
        params,
        experience_grid,
        model_disc,
        model_cont,
        value_disc,
        policy_disc,
        endog_grid_disc,
        value_cont,
        policy_cont,
        endog_grid_cont,
    )


@pytest.mark.parametrize(
    "period, experience, lagged_choice, choice",
    [
        (1, 0, 1, 0),
        (1, 1, 0, 0),
        (2, 1, 0, 1),
        (3, 2, 1, 0),
        (3, 3, 1, 0),
        (4, 4, 0, 0),
        (4, 0, 1, 1),
    ],
)
def test_replication_discrete_versus_continuous_experience(
    period, experience, lagged_choice, choice, test_setup
):

    (
        params,
        experience_grid,
        model_disc,
        model_cont,
        value_disc,
        policy_disc,
        endog_grid_disc,
        value_cont,
        policy_cont,
        endog_grid_cont,
    ) = test_setup

    exp_share_to_test = experience / period if period > 0 else 0

    state_choice_disc_dict = {
        "period": period,
        "lagged_choice": lagged_choice,
        "experience": experience,
        "dummy_exog": 0,
        "choice": choice,
    }
    state_choice_cont_dict = {
        "period": period,
        "lagged_choice": lagged_choice,
        "dummy_exog": 0,
        "choice": choice,
    }

    idx_state_choice_disc = model_disc["model_structure"]["map_state_choice_to_index"][
        state_choice_disc_dict["period"],
        state_choice_disc_dict["lagged_choice"],
        state_choice_disc_dict["experience"],
        state_choice_disc_dict["dummy_exog"],
        state_choice_disc_dict["choice"],
    ]
    idx_state_choice_cont = model_cont["model_structure"]["map_state_choice_to_index"][
        state_choice_cont_dict["period"],
        state_choice_cont_dict["lagged_choice"],
        state_choice_cont_dict["dummy_exog"],
        state_choice_cont_dict["choice"],
    ]

    # =================================================================================
    # Interpolate
    # =================================================================================

    for wealth_to_test in np.arange(5, 100, 5, dtype=float):

        policy_cont_interp, value_cont_interp = (
            interp2d_policy_and_value_on_wealth_and_regular_grid(
                regular_grid=experience_grid,
                wealth_grid=endog_grid_cont[idx_state_choice_cont],
                policy_grid=policy_cont[idx_state_choice_cont],
                value_grid=value_cont[idx_state_choice_cont],
                regular_point_to_interp=exp_share_to_test,
                wealth_point_to_interp=jnp.array(wealth_to_test),
                compute_utility=model_cont["model_funcs"]["compute_utility"],
                state_choice_vec=state_choice_cont_dict,
                params=params,
            )
        )

        policy_disc_interp, value_disc_interp = interp1d_policy_and_value_on_wealth(
            wealth=jnp.array(wealth_to_test),
            endog_grid=endog_grid_disc[idx_state_choice_disc],
            policy=policy_disc[idx_state_choice_disc],
            value=value_disc[idx_state_choice_disc],
            compute_utility=model_disc["model_funcs"]["compute_utility"],
            state_choice_vec=state_choice_disc_dict,
            params=params,
        )

        aaae(value_cont_interp, value_disc_interp, decimal=1e-6)
        aaae(policy_cont_interp, policy_disc_interp, decimal=1e-6)
