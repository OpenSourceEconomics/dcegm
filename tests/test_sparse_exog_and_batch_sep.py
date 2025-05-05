import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import get_solve_func_for_model, solve_dcegm
from dcegm.toy_models.example_model_functions import load_example_models


def utility_with_exog(consumption, health, partner, params):
    utility_consumption = consumption ** (1 - params["rho"]) / (1 - params["rho"])
    utility_health = (1 - health) * params["health_disutil"]
    utility_partner = partner * params["partner_util"]
    return utility_consumption - utility_health + utility_partner


def health_transition(period, health, params):
    prob_good_health = (
        health * params["good_to_good"] + (1 - health) * params["bad_to_good"]
    )
    # After period 20 you always transition to bad
    prob_good_health = jax.lax.select(period < 20, prob_good_health, 0.0)
    return jnp.array([1 - prob_good_health, prob_good_health])


def partner_transition(period, partner, params):
    prob_married = (1 - partner) * params["single_to_married"] + partner * params[
        "married_to_married"
    ]
    # After period 15 you always transition to married
    prob_married = jax.lax.select(period < 15, prob_married, 1.0)
    return jnp.array([1 - prob_married, prob_married])


def sparsity_condition(period, lagged_choice, health, education, partner):
    # If period is larger than 15 you can not be single

    if period < 20:
        if period > 14 and partner == 0:
            return {
                "period": period,
                "lagged_choice": lagged_choice,
                "education": education,
                "health": health,
                "partner": 1,
            }
        else:
            return True
    else:
        if (health == 0) & (partner == 1):
            return True
        else:
            return {
                "period": period,
                "lagged_choice": lagged_choice,
                "education": education,
                "health": 0,
                "partner": 1,
            }


def test_benchmark_models(load_replication_params_and_specs):
    params, model_specs = load_replication_params_and_specs("retirement_taste_shocks")
    params_update = {
        "health_disutil": 0.1,
        "good_to_good": 0.8,
        "bad_to_good": 0.1,
        "single_to_married": 0.1,
        "married_to_married": 0.9,
    }
    params = {**params, **params_update}

    options = {}

    options["model_params"] = model_specs
    options["model_params"]["n_choices"] = model_specs["n_discrete_choices"]
    options["state_space"] = {
        "n_periods": 25,
        "choices": np.arange(2, dtype=int),
        "endogenous_states": {
            "education": np.arange(2, dtype=int),
        },
        "exogenous_processes": {
            "health": {
                "states": np.arange(2, dtype=int),
                "transition": health_transition,
            },
            "partner": {
                "states": np.arange(2, dtype=int),
                "transition": partner_transition,
            },
        },
        "continuous_states": {
            "wealth": jnp.linspace(
                0,
                options["model_params"]["max_wealth"],
                options["model_params"]["n_grid_points"],
            )
        },
    }

    model_funcs = load_example_models("dcegm_paper")

    model_full = setup_model(
        options=options,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["final_period_utility_functions"],
        budget_constraint=model_funcs["budget_constraint"],
    )

    model_funcs_sparse = model_funcs.copy()
    model_funcs_sparse["sparsity_condition"] = sparsity_condition

    model_sparse = setup_model(
        options=options,
        state_space_functions=model_funcs_sparse["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["final_period_utility_functions"],
        budget_constraint=model_funcs["budget_constraint"],
    )

    value_full, policy_full, endog_grid_full = get_solve_func_for_model(model_full)(
        params
    )
    value_sparse, policy_sparse, endog_grid_sparse = get_solve_func_for_model(
        model_sparse
    )(params)

    state_choices_sparse = model_sparse["model_structure"]["state_choice_space"]
    state_choice_space_tuple_sparse = tuple(
        state_choices_sparse[:, i] for i in range(state_choices_sparse.shape[1])
    )
    full_idxs = model_full["model_structure"]["map_state_choice_to_index_with_proxy"][
        state_choice_space_tuple_sparse
    ]

    aaae(endog_grid_full[full_idxs], endog_grid_sparse)
    aaae(value_full[full_idxs], value_sparse)
    aaae(policy_full[full_idxs], policy_sparse)

    options_sep_once = options.copy()
    options_sep_once["state_space"]["min_period_batch_segments"] = 20

    value_sep_1, policy_sep_1, endog_grid_sep_1 = solve_dcegm(
        params=params,
        options=options_sep_once,
        state_space_functions=model_funcs_sparse["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["final_period_utility_functions"],
        budget_constraint=model_funcs["budget_constraint"],
    )

    aaae(endog_grid_full[full_idxs], endog_grid_sep_1)
    aaae(value_full[full_idxs], value_sep_1)
    aaae(policy_full[full_idxs], policy_sep_1)

    options_sep_twice = options.copy()
    options_sep_twice["state_space"]["min_period_batch_segments"] = [15, 20]

    value_sep_2, policy_sep_2, endog_grid_sep_2 = solve_dcegm(
        params=params,
        options=options_sep_twice,
        state_space_functions=model_funcs_sparse["state_space_functions"],
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["final_period_utility_functions"],
        budget_constraint=model_funcs["budget_constraint"],
    )

    aaae(endog_grid_full[full_idxs], endog_grid_sep_2)
    aaae(value_full[full_idxs], value_sep_2)
    aaae(policy_full[full_idxs], policy_sep_2)
