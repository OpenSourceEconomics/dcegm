"""Test module for EGM steps:

- aggregate_marg_utils_and_exp_values
- calculate_candidate_solutions_from_euler_equation

"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from dcegm.egm.aggregate_marginal_utility import aggregate_marg_utils_and_exp_values
from dcegm.egm.solve_euler_equation import (
    calculate_candidate_solutions_from_euler_equation,
)
from dcegm.pre_processing.exog_processes import create_exog_transition_function
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options
from tests.two_period_models.model import prob_exog_ltc
from toy_models.consumption_retirement_model.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)

WEALTH_GRID_POINTS = 100

PARAMS = {
    "rho": 0.5,
    "delta": 0.5,
    "interest_rate": 0.02,
    "ltc_cost": 5,
    "wage_avg": 8,
    "sigma": 1,
    "lambda": 10,
    "beta": 0.95,
    # Exogenous parameters
    "ltc_prob_constant": 0.3,
    "ltc_prob_age": 0.1,
    "job_offer_constant": 0.5,
    "job_offer_age": 0,
    "job_offer_educ": 0,
    "job_offer_type_two": 0.4,
}

OPTIONS = {
    "model_params": {
        "n_grid_points": WEALTH_GRID_POINTS,
        "max_wealth": 50,
        "quadrature_points_stochastic": 5,
        "n_choices": 2,
    },
    "state_space": {
        "n_periods": 2,
        "choices": np.arange(2),
        "endogenous_states": {
            "married": [0, 1],
        },
        "exogenous_processes": {
            "ltc": {"transition": prob_exog_ltc, "states": [0, 1]},
        },
    },
}


@pytest.fixture()
def input_for_aggregation():

    key = random.PRNGKey(0)

    taste_shock_scale = 1
    income_shock_weights = jnp.array(
        [0.11846344, 0.23931434, 0.28444444, 0.23931434, 0.11846344]
    )
    states_to_choices_child_states = jnp.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
            [65311, 65311, 24],
            [65311, 65311, 25],
            [65311, 65311, 26],
            [65311, 65311, 27],
            [28, 29, 30],
            [31, 32, 33],
            [34, 35, 36],
            [37, 38, 39],
            [40, 41, 42],
            [43, 44, 45],
            [46, 47, 48],
            [49, 50, 51],
            [65311, 65311, 52],
            [65311, 65311, 53],
            [65311, 65311, 54],
            [65311, 65311, 55],
            [56, 57, 58],
            [59, 60, 61],
            [62, 63, 64],
            [65, 66, 67],
            [68, 69, 70],
            [71, 72, 73],
            [74, 75, 76],
            [77, 78, 79],
            [65311, 65311, 80],
            [65311, 65311, 81],
            [65311, 65311, 82],
            [65311, 65311, 83],
            [84, 85, 86],
            [87, 88, 89],
            [90, 91, 92],
            [93, 94, 95],
            [96, 97, 98],
            [99, 100, 101],
            [102, 103, 104],
            [105, 106, 107],
            [65311, 65311, 108],
            [65311, 65311, 109],
            [65311, 65311, 110],
            [65311, 65311, 111],
            [112, 113, 114],
            [115, 116, 117],
            [118, 119, 120],
            [121, 122, 123],
            [124, 125, 126],
            [127, 128, 129],
            [130, 131, 132],
            [133, 134, 135],
            [65311, 65311, 136],
            [65311, 65311, 137],
            [65311, 65311, 138],
            [65311, 65311, 139],
        ],
        dtype=np.uint16,
    )

    base_array = jnp.linspace(0, 100, 100)  # Shape (100,)
    variation = random.uniform(
        key, shape=(140, 1), minval=0, maxval=20
    )  # Shape (140, 1)

    varied_array = base_array + variation  # Shape (140, 100)

    value_interp = jnp.expand_dims(varied_array, axis=-1)  # Shape (140, 100, 1)
    value_interp = jnp.tile(value_interp, (1, 1, 5))  # Shape (140, 100, 5)

    marg_util_interp = value_interp.copy()

    return (
        value_interp,
        marg_util_interp,
        states_to_choices_child_states,
        taste_shock_scale,
        income_shock_weights,
    )


@pytest.fixture()
def test_input_for_euler_equation():
    model_params_options = OPTIONS["model_params"]
    compute_exog_transition_vec, _ = create_exog_transition_function(OPTIONS)

    utility_functions = create_utility_function_dict()

    compute_utility = determine_function_arguments_and_partial_options(
        func=utility_functions["utility"], options=model_params_options
    )
    compute_inverse_marginal_utility = determine_function_arguments_and_partial_options(
        func=utility_functions["inverse_marginal_utility"],
        options=model_params_options,
    )
    model_funcs = {
        "compute_utility": compute_utility,
        "compute_inverse_marginal_utility": compute_inverse_marginal_utility,
        "compute_exog_transition_vec": compute_exog_transition_vec,
    }

    exog_savings_grid = jnp.linspace(0, 10_000, 100)

    state_choice_mat = {
        "choice": jnp.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype=np.uint8),
        "lagged_choice": jnp.array(
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.uint8
        ),
        "ltc": jnp.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8),
        "married": jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.uint8),
        "period": jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
    }

    child_state_idxs = jnp.array(
        [
            [0, 1],
            [2, 3],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3],
            [4, 5],
            [6, 7],
            [4, 5],
            [6, 7],
            [6, 7],
            [6, 7],
        ],
        dtype=np.uint8,
    )

    return (
        exog_savings_grid,
        state_choice_mat,
        child_state_idxs,
        model_funcs,
    )


def test_aggregation_1d(input_for_aggregation):

    (
        value_interp,
        marg_util_interp,
        states_to_choices_child_states,
        taste_shock_scale,
        income_shock_weights,
    ) = input_for_aggregation

    marg_util, emax = aggregate_marg_utils_and_exp_values(
        value_state_choice_specific=value_interp,
        marg_util_state_choice_specific=marg_util_interp,
        reshape_state_choice_vec_to_mat=states_to_choices_child_states,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
    )

    np.testing.assert_equal(marg_util.shape, (60, 100))
    np.testing.assert_equal(emax.shape, (60, 100))


def test_aggregation_2d(input_for_aggregation):

    (
        value_interp1d,
        marg_util_interp1d,
        states_to_choices_child_states,
        taste_shock_scale,
        income_shock_weights,
    ) = input_for_aggregation

    _value_interp2d = jnp.expand_dims(value_interp1d, axis=1)  # Shape (140, 1, 100, 5)
    value_interp2d = jnp.tile(_value_interp2d, (1, 6, 1, 1))  # Shape (140, 6, 100, 5)

    _marg_util_interp2d = jnp.expand_dims(
        marg_util_interp1d, axis=1
    )  # Shape (140, 1, 100, 5)
    marg_util_interp2d = jnp.tile(
        _marg_util_interp2d, (1, 6, 1, 1)
    )  # Shape (140, 6, 100, 5)

    marg_util_2d, emax_2d = aggregate_marg_utils_and_exp_values(
        value_state_choice_specific=value_interp2d,
        marg_util_state_choice_specific=marg_util_interp2d,
        reshape_state_choice_vec_to_mat=states_to_choices_child_states,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
    )

    np.testing.assert_equal(marg_util_2d.shape, (60, 6, 100))
    np.testing.assert_equal(emax_2d.shape, (60, 6, 100))


def test_euler_1d(test_input_for_euler_equation):

    (
        exog_savings_grid,
        state_choice_mat,
        child_state_idxs,
        model_funcs,
    ) = test_input_for_euler_equation

    key = random.PRNGKey(42)

    key, subkey = random.split(key)
    marg_util = random.uniform(subkey, shape=(8, 100), minval=0.0, maxval=100.0)

    key, subkey = random.split(key)
    emax = random.uniform(subkey, shape=(8, 100), minval=0.0, maxval=100.0)

    (
        endog_grid_candidate,
        value_candidate,
        policy_candidate,
        expected_values,
    ) = calculate_candidate_solutions_from_euler_equation(
        exog_grids={"wealth": exog_savings_grid},
        marg_util_next=marg_util,
        emax_next=emax,
        state_choice_mat=state_choice_mat,
        idx_post_decision_child_states=child_state_idxs,
        model_funcs=model_funcs,
        has_second_continuous_state=False,
        params=PARAMS,
    )

    np.testing.assert_equal(
        endog_grid_candidate.shape,
        (child_state_idxs.shape[0], exog_savings_grid.shape[0]),
    )
    np.testing.assert_equal(
        value_candidate.shape,
        (child_state_idxs.shape[0], exog_savings_grid.shape[0]),
    )
    np.testing.assert_equal(
        policy_candidate.shape,
        (child_state_idxs.shape[0], exog_savings_grid.shape[0]),
    )
    np.testing.assert_equal(
        expected_values.shape,
        (child_state_idxs.shape[0], exog_savings_grid.shape[0]),
    )


def test_euler_2d(test_input_for_euler_equation):

    n_continuous_state = 7

    (
        exog_savings_grid,
        state_choice_mat,
        child_state_idxs,
        model_funcs,
    ) = test_input_for_euler_equation

    key = random.PRNGKey(42)

    key, subkey = random.split(key)
    marg_util = random.uniform(
        subkey,
        shape=(8, n_continuous_state, exog_savings_grid.shape[0]),
        minval=0.0,
        maxval=100.0,
    )

    key, subkey = random.split(key)
    emax = random.uniform(
        subkey,
        shape=(8, n_continuous_state, exog_savings_grid.shape[0]),
        minval=0.0,
        maxval=100.0,
    )

    exog_grids = {
        "wealth": exog_savings_grid,
        "second_continuous": jnp.linspace(0, 1, 7),
    }

    (
        endog_grid_candidate,
        value_candidate,
        policy_candidate,
        expected_values,
    ) = calculate_candidate_solutions_from_euler_equation(
        exog_grids=exog_grids,
        marg_util_next=marg_util,
        emax_next=emax,
        state_choice_mat=state_choice_mat,
        idx_post_decision_child_states=child_state_idxs,
        model_funcs=model_funcs,
        has_second_continuous_state=True,
        params=PARAMS,
    )

    np.testing.assert_equal(
        endog_grid_candidate.shape,
        (child_state_idxs.shape[0], n_continuous_state, exog_savings_grid.shape[0]),
    )
    np.testing.assert_equal(
        value_candidate.shape,
        (child_state_idxs.shape[0], n_continuous_state, exog_savings_grid.shape[0]),
    )
    np.testing.assert_equal(
        policy_candidate.shape,
        (child_state_idxs.shape[0], n_continuous_state, exog_savings_grid.shape[0]),
    )
    np.testing.assert_equal(
        expected_values.shape,
        (child_state_idxs.shape[0], n_continuous_state, exog_savings_grid.shape[0]),
    )
