# """Test module for exogenous processes."""

# import copy
# from itertools import product

# import jax.numpy as jnp
# import numpy as np
# import pytest
# from numpy.testing import assert_almost_equal as aaae

# from dcegm.interface import validate_stochastic_states
# from dcegm.pre_processing.check_options import check_options_and_set_defaults
# from dcegm.pre_processing.model_functions import process_model_functions
# from dcegm.pre_processing.model_structure.stochastic_states import (
#     create_stochastic_states_mapping,
# )
# from dcegm.pre_processing.model_structure.model_structure import create_model_structure
# from dcegm.pre_processing.setup_model import setup_model
# from toy_models.cons_ret_model_dcegm_paper.budget_constraint import budget_constraint
# from toy_models.cons_ret_model_dcegm_paper.state_space_objects import (
#     create_state_space_function_dict,
# )
# from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
#     create_final_period_utility_function_dict,
#     create_utility_function_dict,
# )


# def trans_prob_care_demand(health_state, params):
#     prob_care_demand = (
#         (health_state == 0) * params["care_demand_good_health"]
#         + (health_state == 1) * params["care_demand_medium_health"]
#         + (health_state == 2) * params["care_demand_bad_health"]
#     )

#     return prob_care_demand


# def prob_exog_health_father(health_mother, params):
#     prob_good_health = (
#         (health_mother == 0) * 0.7
#         + (health_mother == 1) * 0.3
#         + (health_mother == 2) * 0.2
#     )
#     prob_medium_health = (
#         (health_mother == 0) * 0.2
#         + (health_mother == 1) * 0.5
#         + (health_mother == 2) * 0.2
#     )
#     prob_bad_health = (
#         (health_mother == 0) * 0.1
#         + (health_mother == 1) * 0.2
#         + (health_mother == 2) * 0.6
#     )

#     return jnp.array([prob_good_health, prob_medium_health, prob_bad_health])


# def prob_exog_health_mother(health_father, params):
#     prob_good_health = (
#         (health_father == 0) * 0.7
#         + (health_father == 1) * 0.3
#         + (health_father == 2) * 0.2
#     )
#     prob_medium_health = (
#         (health_father == 0) * 0.2
#         + (health_father == 1) * 0.5
#         + (health_father == 2) * 0.2
#     )
#     prob_bad_health = (
#         (health_father == 0) * 0.1
#         + (health_father == 1) * 0.2
#         + (health_father == 2) * 0.6
#     )

#     return jnp.array([prob_good_health, prob_medium_health, prob_bad_health])


# def prob_exog_health_child(health_child, params):
#     prob_good_health = (health_child == 0) * 0.7 + (health_child == 1) * 0.1
#     prob_medium_health = (health_child == 0) * 0.3 + (health_child == 1) * 0.9

#     return jnp.array([prob_good_health, prob_medium_health])


# def prob_exog_health_grandma(health_grandma, params):
#     prob_good_health = (health_grandma == 0) * 0.8 + (health_grandma == 1) * 0.15
#     prob_medium_health = (health_grandma == 0) * 0.2 + (health_grandma == 1) * 0.85

#     return jnp.array([prob_good_health, prob_medium_health])


# EXOG_STATE_GRID = [0, 1, 2]
# EXOG_STATE_GRID_SMALL = [0, 1]


# @pytest.mark.parametrize(
#     "health_state_mother, health_state_father, health_state_child, health_state_grandma",
#     product(
#         EXOG_STATE_GRID, EXOG_STATE_GRID, EXOG_STATE_GRID_SMALL, EXOG_STATE_GRID_SMALL
#     ),
# )
# def test_exog_processes(
#     health_state_mother, health_state_father, health_state_child, health_state_grandma
# ):
#     params = {
#         "rho": 0.5,
#         "delta": 0.5,
#         "interest_rate": 0.02,
#         "ltc_cost": 5,
#         "wage_avg": 8,
#         "sigma": 1,
#         "taste_shock_scale": 1,
#         "ltc_prob": 0.3,
#         "beta": 0.95,
#     }

#     options = {
#         "model_params": {
#             "quadrature_points_stochastic": 5,
#             "n_choices": 2,
#         },
#         "state_space": {
#             "n_periods": 2,
#             "choices": np.arange(2),
#             "deterministic_states": {
#                 "married": [0, 1],
#             },
#             "continuous_states": {
#                 "assets_end_of_period": np.linspace(0, 50, 100),
#             },
#             "stochastic_states": {
#                 "health_mother": {
#                     "transition": prob_exog_health_mother,
#                     "states": [0, 1, 2],
#                 },
#                 "health_father": {
#                     "transition": prob_exog_health_father,
#                     "states": [0, 1, 2],
#                 },
#                 "health_child": {
#                     "transition": prob_exog_health_child,
#                     "states": [0, 1],
#                 },
#                 "health_grandma": {
#                     "transition": prob_exog_health_grandma,
#                     "states": [0, 1],
#                 },
#             },
#         },
#     }

#     options = check_options_and_set_defaults(options)

#     model = setup_model(
#         options,
#         state_space_functions=create_state_space_function_dict(),
#         utility_functions=create_utility_function_dict(),
#         utility_functions_final_period=create_final_period_utility_function_dict(),
#         budget_constraint=budget_constraint,
#     )
#     model_funcs = model["model_funcs"]
#     model_structure = model["model_structure"]

#     stochastic_state_mapping = create_stochastic_states_mapping(
#         model_structure["stochastic_state_space"].astype(np.int16),
#         model_structure["stochastic_states_names"],
#     )

#     # Test the interface validation function for exogenous processes
#     invalid_model = copy.deepcopy(model)
#     with pytest.raises(
#         ValueError, match="does not return float transition probabilities"
#     ):
#         invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
#             lambda **kwargs: jnp.array([1, 3, 4])
#         )  # Returns an array instead of a float
#         validate_stochastic_states(invalid_model, params)

#     with pytest.raises(
#         ValueError, match="does not return non-negative transition probabilities"
#     ):
#         invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
#             lambda **kwargs: jnp.array([0.7, -0.3, 0.6])
#         )  # Contains negative values
#         validate_stochastic_states(invalid_model, params)

#     with pytest.raises(
#         ValueError, match="does not return transition probabilities less or equal to 1"
#     ):
#         invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
#             lambda **kwargs: jnp.array([0.7, 1.3, 0.6])
#         )  # Contains values geq 1
#         validate_stochastic_states(invalid_model, params)

#     with pytest.raises(
#         ValueError, match="does not return the correct number of transitions"
#     ):
#         invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
#             lambda **kwargs: jnp.array([0.7, 0.3])
#         )  # Wrong number of states (only 2 instead of 3)
#         validate_stochastic_states(invalid_model, params)

#     with pytest.raises(ValueError, match="transition probabilities do not sum to 1"):
#         invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
#             lambda **kwargs: jnp.array([0.6, 0.3, 0.2])
#         )  # Doesn't sum to 1
#         validate_stochastic_states(invalid_model, params)

#     # Check if valid model passes
#     assert validate_stochastic_states(model, params)

#     # Check if mapping works
#     mother_bad_health = np.where(model_structure["stochastic_state_space"][:, 0] == 2)[0]

#     for exog_state in mother_bad_health:
#         assert stochastic_state_mapping(exog_proc_state=exog_state)["health_mother"] == 2

#     # Now check probabilities
#     state_choices_test = {
#         "period": 0,
#         "lagged_choice": 0,
#         "married": 0,
#         "health_mother": health_state_mother,
#         "health_father": health_state_father,
#         "health_child": health_state_child,
#         "health_grandma": health_state_grandma,
#         "choice": 0,
#     }
#     prob_vector = model_funcs["compute_stochastic_transition_vec"](
#         params=params, **state_choices_test
#     )
#     prob_mother_health = model_funcs["processed_stochastic_funcs"]["health_mother"](
#         params=params, **state_choices_test
#     )
#     prob_father_health = model_funcs["processed_stochastic_funcs"]["health_father"](
#         params=params, **state_choices_test
#     )
#     prob_child_health = model_funcs["processed_stochastic_funcs"]["health_child"](
#         params=params, **state_choices_test
#     )
#     prob_grandma_health = model_funcs["processed_stochastic_funcs"]["health_grandma"](
#         params=params, **state_choices_test
#     )

#     for exog_val, prob in enumerate(prob_vector):
#         child_prob_states = stochastic_state_mapping(exog_val)
#         prob_mother = prob_mother_health[child_prob_states["health_mother"]]
#         prob_father = prob_father_health[child_prob_states["health_father"]]
#         prob_child = prob_child_health[child_prob_states["health_child"]]
#         prob_grandma = prob_grandma_health[child_prob_states["health_grandma"]]
#         prob_expec = prob_mother * prob_father * prob_child * prob_grandma
#         aaae(prob, prob_expec)


# def test_nested_exog_process():
#     """Tests that nested exogenous transition probs are calculated correctly.

#     >>> 0.3 * 0.8 + 0.3 * 0.7 + 0.4 * 0.6
#     0.69
#     >>> 0.3 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4
#     0.31000000000000005

#     """
#     params = {
#         "care_demand_good_health": 0.2,
#         "care_demand_medium_health": 0.3,
#         "care_demand_bad_health": 0.4,
#     }

#     trans_probs_health = jnp.array([0.3, 0.3, 0.4])

#     prob_care_good = trans_prob_care_demand(health_state=0, params=params)
#     prob_care_medium = trans_prob_care_demand(health_state=1, params=params)
#     prob_care_bad = trans_prob_care_demand(health_state=2, params=params)

#     _trans_probs_care_demand = jnp.array(
#         [prob_care_good, prob_care_medium, prob_care_bad]
#     )
#     joint_trans_prob = trans_probs_health @ _trans_probs_care_demand
#     expected = 0.3 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4

#     aaae(joint_trans_prob, expected)

"""Test module for exogenous processes."""

import copy
from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_almost_equal as aaae

from dcegm.interfaces.interface import validate_stochastic_transition
from dcegm.pre_processing.model_structure.stochastic_states import (
    create_stochastic_state_mapping,
)
from dcegm.pre_processing.setup_model import create_model_dict
from dcegm.toy_models.cons_ret_model_dcegm_paper import (
    budget_constraint,
    create_final_period_utility_function_dict,
    create_state_space_function_dict,
    create_utility_function_dict,
)


def trans_prob_care_demand(health_state, params):
    """
    Compute the probability of needing care, conditional on health state.

    Args:
        health_state (int): Health state indicator (0, 1, or 2).
        params (dict): Model parameters that include probabilities for different
            health states.

    Returns:
        float: Probability of needing care in the current period.
    """
    prob_care_demand = (
        (health_state == 0) * params["care_demand_good_health"]
        + (health_state == 1) * params["care_demand_medium_health"]
        + (health_state == 2) * params["care_demand_bad_health"]
    )
    return prob_care_demand


def prob_exog_health_father(health_mother, params):
    """
    Compute transition probabilities for father's health, given mother's health.

    Args:
        health_mother (int): Mother's health state indicator.
        params (dict): Model parameters.

    Returns:
        jnp.ndarray: Probability distribution over father's possible health states.
    """
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
    """
    Compute transition probabilities for mother's health, given father's health.

    Args:
        health_father (int): Father's health state indicator.
        params (dict): Model parameters.

    Returns:
        jnp.ndarray: Probability distribution over mother's possible health states.
    """
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
    """
    Compute transition probabilities for a child's health.

    Args:
        health_child (int): Child's health state indicator.
        params (dict): Model parameters.

    Returns:
        jnp.ndarray: Probability distribution over child's possible health states.
    """
    prob_good_health = (health_child == 0) * 0.7 + (health_child == 1) * 0.1
    prob_medium_health = (health_child == 0) * 0.3 + (health_child == 1) * 0.9

    return jnp.array([prob_good_health, prob_medium_health])


def prob_exog_health_grandma(health_grandma, params):
    """
    Compute transition probabilities for a grandmother's health.

    Args:
        health_grandma (int): Grandmother's health state indicator.
        params (dict): Model parameters.

    Returns:
        jnp.ndarray: Probability distribution over grandmother's health states.
    """
    prob_good_health = (health_grandma == 0) * 0.8 + (health_grandma == 1) * 0.15
    prob_medium_health = (health_grandma == 0) * 0.2 + (health_grandma == 1) * 0.85

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
    """
    Test consistency of exogenous transition processes with various health states.
    """
    params = {
        "rho": 0.5,
        "delta": 0.5,
        "interest_rate": 0.02,
        "ltc_cost": 5,
        "wage_avg": 8,
        "sigma": 1,
        "taste_shock_scale": 1,
        "ltc_prob": 0.3,
        "beta": 0.95,
    }

    model_specs = {
        "n_choices": 2,
    }
    model_config = {
        "n_quad_points": 5,
        "n_periods": 2,
        "choices": np.arange(2),
        "deterministic_states": {
            "married": [0, 1],
        },
        "continuous_states": {
            "assets_end_of_period": np.linspace(0, 50, 100),
        },
        "stochastic_states": {
            "health_mother": [0, 1, 2],
            "health_father": [0, 1, 2],
            "health_child": [0, 1],
            "health_grandma": [0, 1],
        },
    }

    stochastic_state_transitions = {
        "health_mother": prob_exog_health_mother,
        "health_father": prob_exog_health_father,
        "health_child": prob_exog_health_child,
        "health_grandma": prob_exog_health_grandma,
    }

    model = create_model_dict(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_constraint,
        stochastic_states_transitions=stochastic_state_transitions,
    )
    model_funcs = model["model_funcs"]
    model_structure = model["model_structure"]

    stochastic_state_mapping = create_stochastic_state_mapping(
        model_structure["stochastic_state_space"].astype(np.int16),
        model_structure["stochastic_states_names"],
    )

    # Test the interface validation function for exogenous processes
    invalid_model = copy.deepcopy(model)
    with pytest.raises(
        ValueError, match="does not return float transition probabilities"
    ):
        invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
            lambda **kwargs: jnp.array([1, 3, 4])
        )
        validate_stochastic_transition(invalid_model, params)

    with pytest.raises(
        ValueError, match="returns one or more negative transition probabilities"
    ):
        invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
            lambda **kwargs: jnp.array([0.7, -0.3, 0.6])
        )
        validate_stochastic_transition(invalid_model, params)

    with pytest.raises(
        ValueError, match="returns one or more transition probabilities > 1"
    ):
        invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
            lambda **kwargs: jnp.array([0.7, 1.3, 0.6])
        )
        validate_stochastic_transition(invalid_model, params)

    with pytest.raises(
        ValueError, match="does not return the correct number of transitions"
    ):
        invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
            lambda **kwargs: jnp.array([0.7, 0.3])
        )
        validate_stochastic_transition(invalid_model, params)

    with pytest.raises(ValueError, match="transition probabilities do not sum to 1"):
        invalid_model["model_funcs"]["processed_stochastic_funcs"]["health_mother"] = (
            lambda **kwargs: jnp.array([0.6, 0.3, 0.2])
        )
        validate_stochastic_transition(invalid_model, params)

    # Check if valid model passes
    assert validate_stochastic_transition(model, params)

    # Check if mapping works
    mother_bad_health = np.where(model_structure["stochastic_state_space"][:, 0] == 2)[
        0
    ]
    for idx in mother_bad_health:
        assert stochastic_state_mapping(state_idx=idx)["health_mother"] == 2

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

    prob_vector = model_funcs["compute_stochastic_transition_vec"](
        params=params, **state_choices_test
    )
    prob_mother_health = model_funcs["processed_stochastic_funcs"]["health_mother"](
        params=params, **state_choices_test
    )
    prob_father_health = model_funcs["processed_stochastic_funcs"]["health_father"](
        params=params, **state_choices_test
    )
    prob_child_health = model_funcs["processed_stochastic_funcs"]["health_child"](
        params=params, **state_choices_test
    )
    prob_grandma_health = model_funcs["processed_stochastic_funcs"]["health_grandma"](
        params=params, **state_choices_test
    )

    for exog_val, prob in enumerate(prob_vector):
        child_prob_states = stochastic_state_mapping(exog_val)
        prob_mother = prob_mother_health[child_prob_states["health_mother"]]
        prob_father = prob_father_health[child_prob_states["health_father"]]
        prob_child = prob_child_health[child_prob_states["health_child"]]
        prob_grandma = prob_grandma_health[child_prob_states["health_grandma"]]
        prob_expec = prob_mother * prob_father * prob_child * prob_grandma

        aaae(prob, prob_expec)


def test_nested_exog_process():
    """
    Test that nested exogenous transition probabilities are computed correctly.

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
