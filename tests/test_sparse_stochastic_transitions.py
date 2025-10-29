"""Test sparse stochastic transitions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
from dcegm.toy_models.cons_ret_model_dcegm_paper import (
    budget_constraint,
    create_final_period_utility_function_dict,
    create_state_space_function_dict,
    create_utility_function_dict,
)


def prob_exog_health_father(health_mother):
    """Sparse transition: can only go to 2 out of 3 states depending on mother's health."""
    # health_mother == 0: can go to states 0, 1 (not 2)
    # health_mother == 1: can go to states 1, 2 (not 0)
    # health_mother == 2: can go to states 0, 2 (not 1)
    prob_good_health = (
        (health_mother == 0) * 0.7
        + (health_mother == 1) * 0.0
        + (health_mother == 2) * 0.3
    )
    prob_medium_health = (
        (health_mother == 0) * 0.3
        + (health_mother == 1) * 0.6
        + (health_mother == 2) * 0.0
    )
    prob_bad_health = (
        (health_mother == 0) * 0.0
        + (health_mother == 1) * 0.4
        + (health_mother == 2) * 0.7
    )
    return jnp.array([prob_good_health, prob_medium_health, prob_bad_health])


def prob_exog_health_mother(health_father):
    """Sparse transition: can only go to 2 out of 3 states depending on father's health."""
    # health_father == 0: can go to states 0, 1 (not 2)
    # health_father == 1: can go to states 0, 2 (not 1)
    # health_father == 2: can go to states 1, 2 (not 0)
    prob_good_health = (
        (health_father == 0) * 0.8
        + (health_father == 1) * 0.4
        + (health_father == 2) * 0.0
    )
    prob_medium_health = (
        (health_father == 0) * 0.2
        + (health_father == 1) * 0.0
        + (health_father == 2) * 0.3
    )
    prob_bad_health = (
        (health_father == 0) * 0.0
        + (health_father == 1) * 0.6
        + (health_father == 2) * 0.7
    )
    return jnp.array([prob_good_health, prob_medium_health, prob_bad_health])


def prob_exog_health_child(health_child, params):
    """Compute transition probabilities for a child's health."""
    prob_good_health = (health_child == 0) * 0.7 + (health_child == 1) * 0.1
    prob_medium_health = (health_child == 0) * 0.3 + (health_child == 1) * 0.9
    return jnp.array([prob_good_health, prob_medium_health])


def prob_exog_health_grandma(health_grandma):
    """Compute transition probabilities for a grandmother's health."""
    # This function has not every state reachable from every other state. This should not be reduced
    # in sparsity.
    prob_good_health = (health_grandma == 0) * 1.0 + (health_grandma == 1) * 0.15
    return jnp.array([prob_good_health, 1 - prob_good_health])


def util_new(
    consumption,
    choice,
    params,
    health_mother,
    health_father,
    health_child,
    health_grandma,
):

    utility_consumption = jax.lax.select(
        jnp.allclose(params["rho"], 1),
        jnp.log(consumption),
        (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"]),
    )

    utility = (
        utility_consumption
        - (1 - choice) * params["delta"]
        + 2 * health_mother
        + 1.5 * health_father
        + health_child
        + 0.5 * health_grandma
        + (1 - health_grandma) * 5
    )

    return utility


def test_sparse_stochastic_transitions():
    """Test that solving with sparse transitions gives same results."""

    params = {
        "rho": 2,
        "delta": 0.5,
        "discount_factor": 0.95,
        "taste_shock_scale": 1,
        "income_shock_std": 1,
        "income_shock_mean": 0.0,
        "interest_rate": 0.05,
        "constant": 1,
        "exp": 0.1,
        "exp_squared": -0.01,
        "consumption_floor": 0.5,
    }

    model_specs = {"n_choices": 2, "taste_shock_scale": 1, "min_age": 20}

    model_config = {
        "n_quad_points": 5,
        "n_periods": 10,
        "choices": np.arange(2),
        "deterministic_states": {
            "married": [0, 1],
        },
        "continuous_states": {
            "assets_end_of_period": np.linspace(0, 50, 100),
        },
        "stochastic_states": {
            "health_mother": [0, 1, 2],
            "health_grandma": [0, 1],
            "health_father": [0, 1, 2],
            "health_child": [0, 1],
        },
    }

    stochastic_state_transitions = {
        "health_mother": prob_exog_health_mother,
        "health_grandma": prob_exog_health_grandma,
        "health_child": prob_exog_health_child,
        "health_father": prob_exog_health_father,
    }

    # Setup model first time
    model_1 = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_constraint,
        stochastic_states_transitions=stochastic_state_transitions,
        use_stochastic_sparsity=False,
    )

    # Setup model second time with same config
    model_2 = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_constraint,
        stochastic_states_transitions=stochastic_state_transitions,
        use_stochastic_sparsity=True,
    )

    # Solve both models
    model_solved_1 = model_1.solve(params=params)
    model_solved_2 = model_2.solve(params=params)

    # Test that solutions are identical
    aaae(model_solved_1.endog_grid, model_solved_2.endog_grid)
    aaae(model_solved_1.policy, model_solved_2.policy)
    aaae(model_solved_1.value, model_solved_2.value)
