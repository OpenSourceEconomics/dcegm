import copy
from itertools import product
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
from toy_models.load_example_model import load_example_models

N_PERIODS = 5
N_DISCRETE_CHOICES = 2
MAX_WEALTH = 50
WEALTH_GRID_POINTS = 100
# If I set GRID points to 61 (5 * 4 * 3 *2 * 1), the test will pass
# on complete precise numbers
EXPERIENCE_GRID_POINTS = 5
MAX_INIT_EXPERIENCE = 1

PARAMS = {
    "beta": 0.95,
    "delta": 0.35,
    "rho": 1.95,
    "interest_rate": 0.04,
    "taste_shock_scale": 1,  # taste shock (scale) parameter
    "sigma": 1,  # shock on labor income, standard deviation
    "constant": 0.75,
    "exp": 0.04,
    "exp_squared": -0.0002,
    "consumption_floor": 0.001,
}

# ====================================================================================
# Test
# ====================================================================================


@pytest.fixture(scope="session")
def test_setup():

    # =================================================================================
    # Discrete experience
    # =================================================================================

    model_funcs_discr_exp = load_example_models("with_exp")

    model_params = {
        "n_choices": N_DISCRETE_CHOICES,
        "n_quad_points_stochastic": 5,
        "n_periods": N_PERIODS,
        "max_init_experience": MAX_INIT_EXPERIENCE,
    }

    state_space_options = {
        "n_periods": N_PERIODS,
        "choices": np.arange(
            N_DISCRETE_CHOICES,
        ),
        "endogenous_states": {
            "experience": np.arange(N_PERIODS + MAX_INIT_EXPERIENCE),
        },
        "continuous_states": {
            "wealth": jnp.linspace(
                0,
                MAX_WEALTH,
                WEALTH_GRID_POINTS,
            )
        },
    }
    options_discrete = {
        "model_params": model_params,
        "state_space": state_space_options,
    }

    model_disc = setup_model(
        options=options_discrete,
        state_space_functions=model_funcs_discr_exp["state_space_functions"],
        utility_functions=model_funcs_discr_exp["utility_functions"],
        utility_functions_final_period=model_funcs_discr_exp[
            "final_period_utility_functions"
        ],
        budget_constraint=model_funcs_discr_exp["budget_constraint"],
    )

    solve_disc = get_solve_func_for_model(model_disc)
    value_disc, policy_disc, endog_grid_disc = solve_disc(PARAMS)

    # =================================================================================
    # Continuous experience
    # =================================================================================

    experience_grid = jnp.linspace(0, 1, EXPERIENCE_GRID_POINTS)

    options_cont = copy.deepcopy(options_discrete)
    options_cont["state_space"]["continuous_states"]["experience"] = experience_grid
    options_cont["state_space"].pop("endogenous_states")

    model_funcs_cont_exp = load_example_models("with_cont_exp")

    model_cont = setup_model(
        options=options_cont,
        state_space_functions=model_funcs_cont_exp["state_space_functions"],
        utility_functions=model_funcs_cont_exp["utility_functions"],
        utility_functions_final_period=model_funcs_cont_exp[
            "final_period_utility_functions"
        ],
        budget_constraint=model_funcs_cont_exp["budget_constraint"],
    )

    solve_cont = get_solve_func_for_model(model_cont)
    value_cont, policy_cont, endog_grid_cont = solve_cont(PARAMS)

    return (
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
    product(
        np.arange(N_PERIODS),
        np.arange(N_PERIODS + MAX_INIT_EXPERIENCE),
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
        value_disc,
        policy_disc,
        endog_grid_disc,
        value_cont,
        policy_cont,
        endog_grid_cont,
    ) = test_setup

    exp_share_to_test = experience / (period + MAX_INIT_EXPERIENCE)

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

    idx_state_choice_disc = model_disc["model_structure"][
        "map_state_choice_to_index_with_proxy"
    ][
        state_choice_disc_dict["period"],
        state_choice_disc_dict["lagged_choice"],
        state_choice_disc_dict["experience"],
        state_choice_disc_dict["dummy_exog"],
        state_choice_disc_dict["choice"],
    ]
    idx_state_choice_cont = model_cont["model_structure"][
        "map_state_choice_to_index_with_proxy"
    ][
        state_choice_cont_dict["period"],
        state_choice_cont_dict["lagged_choice"],
        state_choice_cont_dict["dummy_exog"],
        state_choice_cont_dict["choice"],
    ]
    state_specific_choice_set = model_disc["model_funcs"]["state_specific_choice_set"](
        **state_choice_disc_dict
    )
    choice_valid = choice in state_specific_choice_set

    sparsity_condition = model_disc["model_funcs"]["sparsity_condition"]
    state_valid = sparsity_condition(
        period=period,
        experience=experience,
        lagged_choice=lagged_choice,
    )

    if state_valid & choice_valid:

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
                    params=PARAMS,
                )
            )

            policy_disc_interp, value_disc_interp = interp1d_policy_and_value_on_wealth(
                wealth=jnp.array(wealth_to_test),
                endog_grid=endog_grid_disc[idx_state_choice_disc],
                policy=policy_disc[idx_state_choice_disc],
                value=value_disc[idx_state_choice_disc],
                compute_utility=model_disc["model_funcs"]["compute_utility"],
                state_choice_vec=state_choice_disc_dict,
                params=PARAMS,
            )

            aaae(value_cont_interp, value_disc_interp, decimal=3)
            aaae(policy_cont_interp, policy_disc_interp, decimal=3)
