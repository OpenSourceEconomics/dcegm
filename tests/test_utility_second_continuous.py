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
from dcegm.solve import get_solve_func_for_model, solve_dcegm
from toy_models.consumption_retirement_model.utility_functions import (
    inverse_marginal_utility_crra,
    marginal_utility_crra,
    marginal_utility_final_consume_all,
    utility_crra,
    utility_final_consume_all,
)

N_PERIODS = 2
N_DISCRETE_CHOICES = 2
MAX_WEALTH = 50
WEALTH_GRID_POINTS = 100
EXPERIENCE_GRID_POINTS = 6


PARAMS = {
    "beta": 0.95,
    "delta": 0.35,
    "rho": 1.95,
    "exp_util": 0,
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


def utility_crra_with_experience(
    consumption: float,
    experience: float,
    choice: int,
    period: int,
    params: Dict[str, float],
) -> float:
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (jnp.array): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    exp_years = experience * period

    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])

    utility = (
        utility_consumption
        - (1 - choice) * params["delta"]  # disutility of working
        + params["exp_util"] * exp_years  # utility of experience
    )

    return utility


def utility_final_consume_all_with_experience(
    choice: int,
    resources: jnp.array,
    experience: float,
    period: int,
    params: Dict[str, float],
):
    exp_years = experience * period

    util_consumption = (resources ** (1 - params["rho"]) - 1) / (1 - params["rho"])
    util = (
        util_consumption
        - (1 - choice) * params["delta"]
        + params["exp_util"] * exp_years
    )

    return util


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


def get_next_period_experience(period, lagged_choice, experience, options, params):
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

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    # if lagged_choice == 1:
    #     feasible_choice_set = np.array([1])
    # else:
    feasible_choice_set = np.arange(n_choices)

    return feasible_choice_set


# ====================================================================================
# Test
# ====================================================================================


# @pytest.fixture(scope="session")
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
        "continuous_states": {
            "wealth": jnp.linspace(
                0,
                MAX_WEALTH,
                WEALTH_GRID_POINTS,
            ),
            "experience": jnp.linspace(0, 1, EXPERIENCE_GRID_POINTS),
        },
    }

    state_space_functions = {
        "get_next_period_state": get_next_period_discrete_state,
        "update_continuous_state": get_next_period_experience,
        "get_state_specific_feasible_choice_set": get_state_specific_feasible_choice_set,
    }

    # =================================================================================
    # With utility function that does NOT depend on experience
    # =================================================================================

    utility_functions = {
        "utility": utility_crra,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
    }
    utility_functions_final_period = {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }

    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_continuous,
    )

    solve = get_solve_func_for_model(model)
    value, policy, endog_grid, *_ = solve(params)

    # =================================================================================
    # With utility function that depends on experience
    # =================================================================================

    utility_functions_with_experience = {
        "utility": utility_crra_with_experience,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
    }
    utility_functions_with_experience_final_period = {
        "utility": utility_final_consume_all_with_experience,
        "marginal_utility": marginal_utility_final_consume_all,
    }

    model_with_exp_util = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions_with_experience,
        utility_functions_final_period=utility_functions_with_experience_final_period,
        budget_constraint=budget_constraint_continuous,
    )

    solve_with_exp_util = get_solve_func_for_model(model_with_exp_util)
    value_with_exp_util, policy_with_exp_util, endog_grid_with_exp_util, *_ = (
        solve_with_exp_util(params)
    )

    aaae(value, value_with_exp_util)
