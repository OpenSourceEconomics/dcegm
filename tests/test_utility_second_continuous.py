import copy
from itertools import product
from typing import Dict

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from dcegm.pre_processing.setup_model import create_model_dict

N_PERIODS = 5
N_DISCRETE_CHOICES = 2
MAX_WEALTH = 50
WEALTH_GRID_POINTS = 100
MAX_INIT_EXPERIENCE = 1

PARAMS = {
    "beta": 0.95,
    "delta": 0.35,
    "rho": 0.5,
    "exp_util": 0.99,
    "interest_rate": 0.04,
    "taste_shock_scale": 1,  # taste shock (scale) parameter
    "sigma": 1,  # shock on labor income, standard deviation
    "constant": 0.75,
    "exp": 0.04,
    "exp_squared": -0.0002,
    "consumption_floor": 0.5,
}

# =====================================================================================
# Model functions for experience dependent utility
# =====================================================================================

# =====================================================================================
# Discrete experience
# =====================================================================================


def utility_exp(
    consumption: float,
    experience: float,
    choice: int,
    params: Dict[str, float],
) -> float:

    utility_consumption = (consumption ** (1 - params["rho"])) / (1 - params["rho"])

    utility = (
        utility_consumption * params["exp_util"] * jnp.log(experience + 2)
        - (1 - choice) * params["delta"]  # disutility of working
    )
    return utility


def marg_utility_exp(
    consumption: float,
    experience: float,
    params: Dict[str, float],
) -> float:

    return (
        (consumption ** (-params["rho"])) * params["exp_util"] * jnp.log(experience + 2)
    )


def inverse_marg_utility_exp(
    marginal_utility,
    experience,
    params,
):

    return (marginal_utility / (params["exp_util"] * jnp.log(experience + 2))) ** (
        -1 / params["rho"]
    )


def utility_final_consume_all_with_exp(
    choice,
    wealth,
    experience,
    params,
):

    util_consumption = (wealth ** (1 - params["rho"])) / (1 - params["rho"])
    util = (
        util_consumption * params["exp_util"] * jnp.log(experience + 2)
        - (1 - choice) * params["delta"]
    )

    return util


def marginal_utility_final_consume_all_with_exp(
    wealth: jnp.array, experience: float, params: Dict[str, float]
) -> jnp.array:

    return (wealth ** (-params["rho"])) * params["exp_util"] * jnp.log(experience + 2)


# =====================================================================================
# Continuous experience
# =====================================================================================


def utility_cont_exp(
    consumption: float,
    experience: float,
    period: int,
    choice: int,
    params: Dict[str, float],
    model_specs: Dict[str, float],
):
    max_init_experience_period = period + model_specs["max_init_experience"]
    experience_years = experience * max_init_experience_period

    return utility_exp(
        consumption=consumption,
        experience=experience_years,
        choice=choice,
        params=params,
    )


def marginal_utility_cont_exp(
    consumption: float,
    experience: float,
    period: int,
    params: Dict[str, float],
    model_specs: Dict[str, float],
):
    max_init_experience_period = period + model_specs["max_init_experience"]
    experience_years = experience * max_init_experience_period

    return marg_utility_exp(
        consumption=consumption,
        experience=experience_years,
        params=params,
    )


def inverse_marginal_utility_cont_exp(
    marginal_utility: float,
    experience: float,
    period: int,
    params: Dict[str, float],
    model_specs: Dict[str, float],
):
    max_init_experience_period = period + model_specs["max_init_experience"]
    experience_years = experience * max_init_experience_period

    return inverse_marg_utility_exp(
        marginal_utility=marginal_utility,
        experience=experience_years,
        params=params,
    )


def utility_final_consume_all_with_cont_exp(
    choice: int,
    wealth: jnp.array,
    experience: float,
    period: int,
    params: Dict[str, float],
    model_specs: Dict[str, float],
):
    max_init_experience_period = period + model_specs["max_init_experience"]
    experience_years = experience * max_init_experience_period

    return utility_final_consume_all_with_exp(
        choice=choice,
        wealth=wealth,
        experience=experience_years,
        params=params,
    )


def marginal_utility_final_consume_all_with_cont_exp(
    wealth, experience, period, params, model_specs
):
    max_init_experience_period = period + model_specs["max_init_experience"]
    experience_years = experience * max_init_experience_period

    return marginal_utility_final_consume_all_with_exp(
        wealth=wealth,
        experience=experience_years,
        params=params,
    )


# ====================================================================================
# Test
# ====================================================================================


@pytest.fixture(scope="session")
def test_setup():

    # =================================================================================
    # Discrete experience
    # =================================================================================

    model_funcs_discr_exp = toy_models.load_example_model_functions("with_exp")

    # params are actually the same for both models. Just name them params.
    _params, model_specs_disc, model_config_disc = (
        toy_models.load_example_params_model_specs_and_config("with_exp")
    )

    # utility_functions_discrete = {
    #     "utility": utility_exp,
    #     "marginal_utility": marg_utility_exp,
    #     "inverse_marginal_utility": inverse_marg_utility_exp,
    # }

    # utility_functions_final_period_discrete = {
    #     "utility": utility_final_consume_all_with_exp,
    #     "marginal_utility": marginal_utility_final_consume_all_with_exp,
    # }

    model_disc = dcegm.setup_model(
        model_config=model_config_disc,
        model_specs=model_specs_disc,
        **model_funcs_discr_exp,
    )
    model_disc_solved = model_disc.solve(PARAMS)

    # =================================================================================
    # Continuous experience
    # =================================================================================

    model_funcs_cont_exp = toy_models.load_example_model_functions("with_cont_exp")
    _, model_specs_cont, model_config_cont = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )

    # Grid needs to be set very fine. Interpolation on state variables which determine
    # utility might not be the smartest way, but still want the package to do it.
    exp_grid_points = 61

    experience_grid = jnp.linspace(0, 1, exp_grid_points)
    model_config_cont["continuous_states"]["experience"] = experience_grid

    # utility_functions_cont_exp = {
    #     "utility": utility_cont_exp,
    #     "marginal_utility": marginal_utility_cont_exp,
    #     "inverse_marginal_utility": inverse_marginal_utility_cont_exp,
    # }

    # utility_functions_final_period_cont_exp = {
    #     "utility": utility_final_consume_all_with_cont_exp,
    #     "marginal_utility": marginal_utility_final_consume_all_with_cont_exp,
    # }

    # model_cont = create_model_dict(
    #     model_specs=model_specs_cont,
    #     model_config=model_config_cont,
    #     state_space_functions=model_funcs_cont_exp["state_space_functions"],
    #     utility_functions=utility_functions_cont_exp,
    #     utility_functions_final_period=utility_functions_final_period_cont_exp,
    #     budget_constraint=model_funcs_cont_exp["budget_constraint"],
    # )

    model_cont = dcegm.setup_model(
        model_config=model_config_cont,
        model_specs=model_specs_cont,
        **model_funcs_cont_exp,
    )
    model_cont_solved = model_cont.solve(PARAMS)

    return (
        experience_grid,
        model_disc,
        model_cont,
        model_disc_solved,
        model_cont_solved,
    )


@pytest.mark.parametrize(
    "period, experience, lagged_choice, choice",
    product(
        np.arange(N_PERIODS),
        np.arange(N_PERIODS),
        np.arange(N_DISCRETE_CHOICES),
        np.arange(N_DISCRETE_CHOICES),
    ),
)
def test_replication_discrete_versus_continuous_experience(
    period, experience, lagged_choice, choice, test_setup
):

    (
        experience_grid,
        model_disc,
        model_cont,
        model_disc_solved,
        model_cont_solved,
    ) = test_setup
    max_period_exp = period + MAX_INIT_EXPERIENCE

    exp_share_to_test = experience / max_period_exp if max_period_exp > 0 else 0

    state_choice_disc_dict = {
        "period": period,
        "lagged_choice": lagged_choice,
        "experience": experience,
        "dummy_stochastic": 0,
        "choice": choice,
    }
    state_choice_cont_dict = {
        "period": period,
        "lagged_choice": lagged_choice,
        "dummy_stochastic": 0,
        "choice": choice,
    }

    idx_state_choice_disc = model_disc.model_structure[
        "map_state_choice_to_index_with_proxy"
    ][
        state_choice_disc_dict["period"],
        state_choice_disc_dict["lagged_choice"],
        state_choice_disc_dict["experience"],
        state_choice_disc_dict["dummy_stochastic"],
        state_choice_disc_dict["choice"],
    ]
    idx_state_choice_cont = model_cont.model_structure[
        "map_state_choice_to_index_with_proxy"
    ][
        state_choice_cont_dict["period"],
        state_choice_cont_dict["lagged_choice"],
        state_choice_cont_dict["dummy_stochastic"],
        state_choice_cont_dict["choice"],
    ]

    state_specific_choice_set = model_disc.model_funcs["state_specific_choice_set"](
        **state_choice_disc_dict
    )
    choice_valid = choice in state_specific_choice_set

    sparsity_condition = model_disc.model_funcs["sparsity_condition"]
    state_valid = sparsity_condition(
        period=period,
        experience=experience,
        lagged_choice=lagged_choice,
    )

    # ================================================================================
    # Interpolate
    # ================================================================================

    # Unpack for discrete case
    endog_grid_disc = model_disc_solved.endog_grid
    policy_disc = model_disc_solved.policy
    value_disc = model_disc_solved.value

    # Unpack for continuous case
    endog_grid_cont = model_cont_solved.endog_grid
    policy_cont = model_cont_solved.policy
    value_cont = model_cont_solved.value

    if state_valid & choice_valid:

        for wealth_to_test in np.arange(1, 100, 5, dtype=float):

            policy_cont_interp, value_cont_interp = (
                interp2d_policy_and_value_on_wealth_and_regular_grid(
                    regular_grid=experience_grid,
                    wealth_grid=endog_grid_cont[idx_state_choice_cont],
                    policy_grid=policy_cont[idx_state_choice_cont],
                    value_grid=value_cont[idx_state_choice_cont],
                    regular_point_to_interp=exp_share_to_test,
                    wealth_point_to_interp=jnp.array(wealth_to_test),
                    compute_utility=model_cont.model_funcs["compute_utility"],
                    state_choice_vec=state_choice_cont_dict,
                    params=PARAMS,
                )
            )

            policy_disc_interp, value_disc_interp = interp1d_policy_and_value_on_wealth(
                wealth=jnp.array(wealth_to_test),
                endog_grid=endog_grid_disc[idx_state_choice_disc],
                policy=policy_disc[idx_state_choice_disc],
                value=value_disc[idx_state_choice_disc],
                compute_utility=model_disc.model_funcs["compute_utility"],
                state_choice_vec=state_choice_disc_dict,
                params=PARAMS,
            )

            aaae(value_cont_interp, value_disc_interp, decimal=3)
            aaae(policy_cont_interp, policy_disc_interp, decimal=3)
