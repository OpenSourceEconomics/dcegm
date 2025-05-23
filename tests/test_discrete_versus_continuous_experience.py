# If we set grid points to 61 (5 * 4 * 3 * 2 * 1), the test will pass
# on complete precise numbers. Look it example model_config for current number


import copy
from itertools import product

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
    params, model_specs_disc, model_config_disc = (
        toy_models.load_example_params_model_specs_and_config("with_exp")
    )

    model_disc = dcegm.setup_model(
        model_specs=model_specs_disc,
        model_config=model_config_disc,
        state_space_functions=model_funcs_discr_exp["state_space_functions"],
        utility_functions=model_funcs_discr_exp["utility_functions"],
        utility_functions_final_period=model_funcs_discr_exp[
            "utility_functions_final_period"
        ],
        budget_constraint=model_funcs_discr_exp["budget_constraint"],
    )

    model_solved_disc = model_disc.solve(params)

    # =================================================================================
    # Continuous experience
    # =================================================================================

    model_funcs_cont_exp = toy_models.load_example_model_functions("with_cont_exp")
    _, model_specs_cont, model_config_cont = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )

    model_cont = dcegm.setup_model(
        model_config=model_config_cont,
        model_specs=model_specs_cont,
        state_space_functions=model_funcs_cont_exp["state_space_functions"],
        utility_functions=model_funcs_cont_exp["utility_functions"],
        utility_functions_final_period=model_funcs_cont_exp[
            "utility_functions_final_period"
        ],
        budget_constraint=model_funcs_cont_exp["budget_constraint"],
    )

    model_solved_cont = model_cont.solve(params)

    return (
        params,
        model_solved_disc,
        model_solved_cont,
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
        model_solved_disc,
        model_solved_cont,
    ) = test_setup

    model_config_cont = model_solved_cont.model_config

    experience_grid = model_config_cont["continuous_states_info"][
        "second_continuous_grid"
    ]

    exp_share_to_test = experience / (period + MAX_INIT_EXPERIENCE)

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

    model_structure_disc = model_solved_disc.model_structure
    model_funcs_disc = model_solved_disc.model_funcs

    model_structure_cont = model_solved_cont.model_structure
    model_funcs_cont = model_solved_cont.model_funcs

    idx_state_choice_disc = model_structure_disc[
        "map_state_choice_to_index_with_proxy"
    ][
        state_choice_disc_dict["period"],
        state_choice_disc_dict["lagged_choice"],
        state_choice_disc_dict["experience"],
        state_choice_disc_dict["dummy_stochastic"],
        state_choice_disc_dict["choice"],
    ]
    idx_state_choice_cont = model_structure_cont[
        "map_state_choice_to_index_with_proxy"
    ][
        state_choice_cont_dict["period"],
        state_choice_cont_dict["lagged_choice"],
        state_choice_cont_dict["dummy_stochastic"],
        state_choice_cont_dict["choice"],
    ]
    state_specific_choice_set = model_funcs_disc["state_specific_choice_set"](
        **state_choice_disc_dict
    )
    choice_valid = choice in state_specific_choice_set

    sparsity_condition = model_funcs_disc["sparsity_condition"]
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
                    wealth_grid=model_solved_cont.endog_grid[idx_state_choice_cont],
                    policy_grid=model_solved_cont.policy[idx_state_choice_cont],
                    value_grid=model_solved_cont.value[idx_state_choice_cont],
                    regular_point_to_interp=exp_share_to_test,
                    wealth_point_to_interp=jnp.array(wealth_to_test),
                    compute_utility=model_funcs_cont["compute_utility"],
                    state_choice_vec=state_choice_cont_dict,
                    params=params,
                    discount_factor=params["discount_factor"],
                )
            )

            policy_disc_interp, value_disc_interp = interp1d_policy_and_value_on_wealth(
                wealth=jnp.array(wealth_to_test),
                endog_grid=model_solved_disc.endog_grid[idx_state_choice_disc],
                policy=model_solved_disc.policy[idx_state_choice_disc],
                value=model_solved_disc.value[idx_state_choice_disc],
                compute_utility=model_funcs_disc["compute_utility"],
                state_choice_vec=state_choice_disc_dict,
                params=params,
                discount_factor=params["discount_factor"],
            )

            aaae(value_cont_interp, value_disc_interp, decimal=3)
            aaae(policy_cont_interp, policy_disc_interp, decimal=3)
