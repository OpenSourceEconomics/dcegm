import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models


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


def test_benchmark_models():
    params, model_specs, _ = toy_models.load_example_params_model_specs_and_config(
        "dcegm_paper_retirement_with_shocks"
    )
    params_update = {
        "health_disutil": 0.1,
        "good_to_good": 0.8,
        "bad_to_good": 0.1,
        "single_to_married": 0.1,
        "married_to_married": 0.9,
    }
    params = {**params, **params_update}

    model_config = {
        "n_periods": 25,
        "choices": np.arange(2, dtype=int),
        "deterministic_states": {
            "education": np.arange(2, dtype=int),
        },
        "stochastic_states": {
            "health": [0, 1],
            "partner": [0, 1],
        },
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(
                0,
                50,
                500,
            )
        },
        "n_quad_points": 5,
    }

    model_funcs = toy_models.load_example_model_functions("dcegm_paper")
    model_funcs["stochastic_states_transitions"] = {
        "health": health_transition,
        "partner": partner_transition,
    }

    model_full = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )

    model_funcs_sparse = model_funcs.copy()
    model_funcs_sparse["state_space_functions"][
        "sparsity_condition"
    ] = sparsity_condition

    model_sparse = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs_sparse
    )

    model_solved_full = model_full.solve(
        params=params,
    )

    model_solved_sparse = model_sparse.solve(
        params=params,
    )

    state_choices_sparse = model_sparse.model_structure["state_choice_space"]
    discrete_states_names = model_full.model_structure["discrete_states_names"]
    states_dict = {
        state: state_choices_sparse[:, id]
        for id, state in enumerate(discrete_states_names + ["choice"])
    }
    (endog_grid_full, policy_full, value_full) = (
        model_solved_full.get_solution_for_discrete_state_choice(
            states=states_dict, choice=state_choices_sparse[:, -1]
        )
    )

    # state_choice_space_tuple_sparse = tuple(
    #     state_choices_sparse[:, i] for i in range(state_choices_sparse.shape[1])
    # )
    # full_idxs = model_solved_full.model_structure[
    #     "map_state_choice_to_index_with_proxy"
    # ][state_choice_space_tuple_sparse]

    aaae(endog_grid_full, model_solved_sparse.endog_grid)
    aaae(policy_full, model_solved_sparse.value)
    aaae(value_full, model_solved_sparse.policy)

    model_config_sep_once = model_config.copy()
    model_config_sep_once["min_period_batch_segments"] = [20]

    model_split = dcegm.setup_model(
        model_config=model_config_sep_once,
        model_specs=model_specs,
        **model_funcs_sparse,
    )

    model_solved_split_once = model_split.solve(params)

    aaae(endog_grid_full, model_solved_split_once.endog_grid)
    aaae(policy_full, model_solved_split_once.value)
    aaae(value_full, model_solved_split_once.policy)

    model_config_split_twice = model_config.copy()
    model_config_split_twice["min_period_batch_segments"] = [15, 20]

    model_split_twice = dcegm.setup_model(
        model_config=model_config_split_twice,
        model_specs=model_specs,
        **model_funcs_sparse,
    )
    model_solved_split_twice = model_split_twice.solve(params)

    aaae(endog_grid_full, model_solved_split_twice.endog_grid)
    aaae(policy_full, model_solved_split_twice.value)
    aaae(value_full, model_solved_split_twice.policy)
