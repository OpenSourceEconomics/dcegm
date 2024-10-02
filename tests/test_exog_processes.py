"""Test module for exogenous processes."""

from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_almost_equal as aaae

from dcegm.pre_processing.exog_processes import create_exog_state_mapping
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import (
    create_discrete_state_space_and_choice_objects,
)
from tests.two_period_models.model import prob_exog_health
from toy_models.cons_ret_model_dcegm_paper.budget_constraint import budget_constraint
from toy_models.cons_ret_model_dcegm_paper.state_space_objects import (
    create_state_space_function_dict,
)
from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)


def trans_prob_care_demand(health_state, params):
    prob_care_demand = (
        (health_state == 0) * params["care_demand_good_health"]
        + (health_state == 1) * params["care_demand_medium_health"]
        + (health_state == 2) * params["care_demand_bad_health"]
    )

    return prob_care_demand


def prob_exog_health_father(health_mother, params):
    prob_good_health = (
        (health_mother == 0) * 0.7
        + (health_mother == 1) * 0.3
        + (health_mother == 2) * 0.2
    )
    prob_medium_health = (
        (health_mother == 0) * 0.2
        + (health_mother == 1) * 0.5
        + (health_mother == 2) * 0.2
    )
    prob_bad_health = (
        (health_mother == 0) * 0.1
        + (health_mother == 1) * 0.2
        + (health_mother == 2) * 0.6
    )

    return jnp.array([prob_good_health, prob_medium_health, prob_bad_health])


def prob_exog_health_mother(health_father, params):
    prob_good_health = (
        (health_father == 0) * 0.7
        + (health_father == 1) * 0.3
        + (health_father == 2) * 0.2
    )
    prob_medium_health = (
        (health_father == 0) * 0.2
        + (health_father == 1) * 0.5
        + (health_father == 2) * 0.2
    )
    prob_bad_health = (
        (health_father == 0) * 0.1
        + (health_father == 1) * 0.2
        + (health_father == 2) * 0.6
    )

    return jnp.array([prob_good_health, prob_medium_health, prob_bad_health])


def prob_exog_health_child(health_child, params):
    prob_good_health = (health_child == 0) * 0.7 + (health_child == 1) * 0.3
    prob_medium_health = (health_child == 0) * 0.2 + (health_child == 1) * 0.5

    return jnp.array([prob_good_health, prob_medium_health])


def prob_exog_health_grandma(health_grandma, params):
    prob_good_health = (health_grandma == 0) * 0.7 + (health_grandma == 1) * 0.3
    prob_medium_health = (health_grandma == 0) * 0.2 + (health_grandma == 1) * 0.5

    return jnp.array([prob_good_health, prob_medium_health])


EXOG_STATE_GRID = [0, 1, 2]
EXOG_STATE_GRID_SMALL = [0, 1]


@pytest.mark.parametrize(
    "health_state_mother, health_state_father, health_state_child, health_state_grandma",
    product(
        EXOG_STATE_GRID, EXOG_STATE_GRID, EXOG_STATE_GRID_SMALL, EXOG_STATE_GRID_SMALL
    ),
)
def test_exog_processes(
    health_state_mother, health_state_father, health_state_child, health_state_grandma
):
    params = {
        "rho": 0.5,
        "delta": 0.5,
        "interest_rate": 0.02,
        "ltc_cost": 5,
        "wage_avg": 8,
        "sigma": 1,
        "lambda": 1,
        "ltc_prob": 0.3,
        "beta": 0.95,
    }

    options = {
        "model_params": {
            "quadrature_points_stochastic": 5,
            "n_choices": 2,
        },
        "state_space": {
            "n_periods": 2,
            "choices": np.arange(2),
            "endogenous_states": {
                "married": [0, 1],
            },
            "continuous_states": {
                "wealth": np.linspace(0, 50, 100),
            },
            "exogenous_processes": {
                "health_mother": {
                    "transition": prob_exog_health_mother,
                    "states": [0, 1, 2],
                },
                "health_father": {
                    "transition": prob_exog_health_father,
                    "states": [0, 1, 2],
                },
                "health_child": {
                    "transition": prob_exog_health_child,
                    "states": [0, 1],
                },
                "health_grandma": {
                    "transition": prob_exog_health_grandma,
                    "states": [0, 1],
                },
            },
        },
    }

    model_funcs = process_model_functions(
        options,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_constraint,
    )
    model_structure = create_discrete_state_space_and_choice_objects(
        options=options,
        model_funcs=model_funcs,
    )

    exog_state_mapping = create_exog_state_mapping(
        model_structure["exog_state_space"].astype(np.int16),
        model_structure["exog_states_names"],
    )
    # First check if mapping works
    mother_bad_health = np.where(model_structure["exog_state_space"][:, 0] == 2)[0]

    for exog_state in mother_bad_health:
        assert exog_state_mapping(exog_proc_state=exog_state)["health_mother"] == 2

    # Now check probabilities
    state_choices_test = {
        "period": 0,
        "lagged_choice": 0,
        "married": 0,
        "health_mother": health_state_mother,
        "health_father": health_state_father,
        "health_child": health_state_child,
        "health_grandma": health_state_grandma,
        "choice": 0,
    }
    prob_vector = model_funcs["compute_exog_transition_vec"](
        params=params, **state_choices_test
    )
    prob_mother_health = model_funcs["processed_exog_funcs"]["health_mother"](
        params=params, **state_choices_test
    )
    prob_father_health = model_funcs["processed_exog_funcs"]["health_father"](
        params=params, **state_choices_test
    )
    prob_child_health = model_funcs["processed_exog_funcs"]["health_child"](
        params=params, **state_choices_test
    )
    prob_grandma_health = model_funcs["processed_exog_funcs"]["health_grandma"](
        params=params, **state_choices_test
    )

    for exog_val, prob in enumerate(prob_vector):
        child_prob_states = exog_state_mapping(exog_val)
        prob_mother = prob_mother_health[child_prob_states["health_mother"]]
        prob_father = prob_father_health[child_prob_states["health_father"]]
        prob_child = prob_child_health[child_prob_states["health_child"]]
        prob_grandma = prob_grandma_health[child_prob_states["health_grandma"]]
        prob_expec = prob_mother * prob_father * prob_child * prob_grandma
        aaae(prob, prob_expec)


def test_nested_exog_process():
    """Tests that nested exogenous transition probs are calculated correctly.

    >>> 0.3 * 0.8 + 0.3 * 0.7 + 0.4 * 0.6
    0.69
    >>> 0.3 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4
    0.31000000000000005

    """
    params = {
        "care_demand_good_health": 0.2,
        "care_demand_medium_health": 0.3,
        "care_demand_bad_health": 0.4,
    }

    trans_probs_health = jnp.array([0.3, 0.3, 0.4])

    prob_care_good = trans_prob_care_demand(health_state=0, params=params)
    prob_care_medium = trans_prob_care_demand(health_state=1, params=params)
    prob_care_bad = trans_prob_care_demand(health_state=2, params=params)

    _trans_probs_care_demand = jnp.array(
        [prob_care_good, prob_care_medium, prob_care_bad]
    )
    joint_trans_prob = trans_probs_health @ _trans_probs_care_demand
    expected = 0.3 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4

    aaae(joint_trans_prob, expected)
