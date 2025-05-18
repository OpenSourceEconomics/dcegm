# If we set grid points to 61 (5 * 4 * 3 * 2 * 1), the test will pass
# on complete precise numbers. Look it example options for current number


import copy
from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm.toy_models as toy_models
from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import get_solve_func_for_model

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
    params, options_discrete = toy_models.load_example_params_model_specs_and_config(
        "with_exp"
    )

    model_disc = setup_model(
        options=options_discrete,
        state_space_functions=model_funcs_discr_exp["state_space_functions"],
        utility_functions=model_funcs_discr_exp["utility_functions"],
        utility_functions_final_period=model_funcs_discr_exp[
            "utility_functions_final_period"
        ],
        budget_constraint=model_funcs_discr_exp["budget_constraint"],
    )

    solve_disc = get_solve_func_for_model(model_disc)
    value_disc, policy_disc, endog_grid_disc = solve_disc(params)

    # =================================================================================
    # Continuous experience
    # =================================================================================

    model_funcs_cont_exp = toy_models.load_example_model_functions("with_cont_exp")
    _, options_cont = toy_models.load_example_params_model_specs_and_config(
        "with_cont_exp"
    )

    model_cont = setup_model(
        options=options_cont,
        state_space_functions=model_funcs_cont_exp["state_space_functions"],
        utility_functions=model_funcs_cont_exp["utility_functions"],
        utility_functions_final_period=model_funcs_cont_exp[
            "utility_functions_final_period"
        ],
        budget_constraint=model_funcs_cont_exp["budget_constraint"],
    )

    solve_cont = get_solve_func_for_model(model_cont)
    value_cont, policy_cont, endog_grid_cont = solve_cont(params)

    return (
        params,
        options_discrete,
        options_cont,
        model_disc,
        model_cont,
        value_disc,
        policy_disc,
        endog_grid_disc,
        value_cont,
        policy_cont,
        endog_grid_cont,
    )


N_PERIODS = 5
N_CHOICES = 2
MAX_INIT_EXPERIENCE = 1


@pytest.mark.parametrize(
    "period, experience, lagged_choice, choice",
    product(
        np.arange(N_PERIODS),
        np.arange(N_PERIODS + MAX_INIT_EXPERIENCE),
        np.arange(N_CHOICES),
        np.arange(N_CHOICES),
    ),
)
def test_replication_discrete_versus_continuous_experience(
    period, experience, lagged_choice, choice, test_setup
):

    (
        params,
        options_discrete,
        options_cont,
        model_disc,
        model_cont,
        value_disc,
        policy_disc,
        endog_grid_disc,
        value_cont,
        policy_cont,
        endog_grid_cont,
    ) = test_setup

    experience_grid = options_cont["state_space"]["continuous_states"]["experience"]

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

            aaae(value_cont_interp, value_disc_interp, decimal=3)
            aaae(policy_cont_interp, policy_disc_interp, decimal=3)
