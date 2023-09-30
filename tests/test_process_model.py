from functools import partial
from pathlib import Path
from typing import Dict

import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.pre_processing.process_model import (
    _get_utility_function_with_filtered_args_and_kwargs,
)
from dcegm.pre_processing.process_model import create_exog_mapping
from dcegm.pre_processing.process_model import get_exog_transition_vec
from dcegm.pre_processing.process_model import process_exog_funcs
from dcegm.pre_processing.process_model import process_params
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.utility_functions import (
    inverse_marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_func_log_crra,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


def func_exog_ltc(
    period,
    lagged_ltc,
    choice,
    params,
    options,
):
    age = period

    prob_ltc = (lagged_ltc == 0) * (
        params["ltc_prob_constant"] + age * params["ltc_prob_age"]
    ) + (lagged_ltc == 1)
    prob_no_ltc = 1 - prob_ltc

    return prob_no_ltc, prob_ltc


def func_exog_job_offer(
    period,
    married,
    lagged_job_offer,
    params,
):
    prob_job_offer = (lagged_job_offer == 0) * params["job_offer_constant"] + (
        lagged_job_offer == 1
    ) * (params["job_offer_constant"] + params["job_offer_type_two"])
    prob_no_job_offer = 1 - prob_job_offer

    return prob_no_job_offer, prob_job_offer


def func_exog_health(age, married, lagged_health, options):
    prob_good_health = (
        (lagged_health == 0) * 0
        + (lagged_health == 1) * 0.3
        + (lagged_health == 2) * 0.7
    )

    prob_medium_health = (
        (lagged_health == 0) * 0
        + (lagged_health == 1) * 0.5
        + (lagged_health == 2) * 0.2
    )

    prob_bad_health = (
        (lagged_health == 0) * 1
        + (lagged_health == 1) * 0.2
        + (lagged_health == 2) * 0.1
    )

    return prob_bad_health, prob_medium_health, prob_good_health


def utility_func_crra_exog(
    consumption: np.array,
    period: int,
    choice: int,
    lagged_ltc: int,
    lagged_job_offer: int,
    params: Dict[str, float],
    options: Dict[str, float],
):
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (jnp.array): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """

    utility_consumption = (consumption ** (1 - params["theta"]) - 1) / (
        1 - params["theta"]
    )

    utility = utility_consumption - (1 - choice) * params["delta"]

    return utility


@pytest.fixture()
def model_setup():
    """Define example options and params."""

    options = {
        "state_space": {
            "exogenous_processes": {
                "ltc": func_exog_ltc,
                "job_offer": func_exog_job_offer,
            },
            "endogenous_states": {
                "period": np.arange(2),
                "married": [0, 1],
                "lagged_choice": [0, 1],
            },
            "exogenous_states": {"lagged_ltc": [0, 1], "lagged_job_offer": [0, 1]},
            "choice": [0, 1],
        },
        "model_params": {"min_age": 50, "max_age": 80},
    }

    params = {
        "theta": 0.3,
        "delta": 0.5,
        "beta": 0,
        "ltc_prob_constant": 0.3,
        "ltc_prob_age": 0.1,
        "job_offer_constant": 0.5,
        "job_offer_age": 0,
        "job_offer_educ": 0,
        "job_offer_type_two": 0.4,
    }

    return options, params


def test_process_utility_funcs(model_setup):
    options, params = model_setup

    exog_mapping = create_exog_mapping(options)
    compute_utility = _get_utility_function_with_filtered_args_and_kwargs(
        utility_func_crra,
        options=options,
        exog_mapping=exog_mapping,
    )

    policy = np.arange(10)

    # for now, exog state is one global exog state: lagged_ltc x lagged_job_offer
    # at index position -2
    state_choice_vec = [0, 0, 0, 0, 1, 10]

    compute_utility(consumption=policy, params=params, *state_choice_vec)


@pytest.mark.parametrize(
    "model",
    [
        ("retirement_no_taste_shocks"),
        ("retirement_taste_shocks"),
        ("deaton"),
    ],
)
def test_missing_parameter(
    model,
    load_example_model,
):
    params, _ = load_example_model(f"{model}")

    indices_to_drop = [
        ("assets", "interest_rate"),
        ("shocks", "sigma"),
        ("shocks", "lambda"),
    ]
    params_missing = params.drop(index=indices_to_drop)

    params_dict = process_params(params_missing)

    for param in ["interest_rate", "sigma", "lambda"]:
        assert param in params_dict.keys()

    params_missing = params_missing.drop(index=("beta", "beta"))
    with pytest.raises(ValueError, match="beta must be provided in params."):
        process_params(params_missing)


@pytest.mark.parametrize(
    "func",
    [
        utility_func_crra_exog,
        utiility_func_log_crra,
        utility_func_crra,
        marginal_utility_crra,
        inverse_marginal_utility_crra,
    ],
)
def test_get_utility_function_with_filtered_args_and_kwargs(func, model_setup):
    options, _ = model_setup

    exog_mapping = create_exog_mapping(options)
    func_with_filtered_args_and_kwargs = (
        _get_utility_function_with_filtered_args_and_kwargs(
            func,
            options=options,
            exog_mapping=exog_mapping,
        )
    )

    state_choice_vec = np.array([10, 1, 9, 2, 2])
    kwargs = {
        "consumption": np.arange(10, 20),
        "choice": 0,
        "marginal_utility": np.arange(1, 11),
        "params": {"theta": 0.5, "delta": 0.1},
        "options": options,
    }

    util = func_with_filtered_args_and_kwargs(*state_choice_vec, **kwargs)

    assert np.allclose(len(util), len(kwargs["consumption"]))


@pytest.mark.skip
def test_recursive_loop():
    np.array(
        [
            [
                [[0.35, 0.35], [0.3, 0.3]],
                [[0.35, 0.35], [0.3, 0.3]],
                [[0.15, 0.15], [0.2, 0.2]],
                [[0.15, 0.15], [0.2, 0.2]],
            ],
            [
                [[0.07, 0.07], [0.06, 0.06]],
                [[0.63, 0.63], [0.54, 0.54]],
                [[0.03, 0.03], [0.04, 0.04]],
                [[0.27, 0.27], [0.36, 0.36]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.9, 0.9], [0.9, 0.9]],
            ],
        ]
    )


def test_exog_mapping():
    options = {
        "state_space": {
            "exogenous_states": {
                "exog_1": [1, 2, 3],
                "exog_2": [4, 5],
                "exog_3": [6, 7, 8, 9],
            }
        }
    }

    # =================================================================================

    exog_state_vars = options["state_space"]["exogenous_states"]

    n_1 = len(exog_state_vars["exog_1"])
    n_2 = len(exog_state_vars["exog_2"])
    n_3 = len(exog_state_vars["exog_3"])

    _expected = []
    for exog_1 in range(n_1):
        for exog_2 in range(n_2):
            for exog_3 in range(n_3):
                _expected += [[exog_1, exog_2, exog_3]]

    expected = np.array(_expected)

    # =================================================================================

    exog_mapping = create_exog_mapping(options)

    aaae(exog_mapping, expected)


def test_get_exog_transition_vec(model_setup):
    options, params = model_setup

    exog_mapping = create_exog_mapping(options)
    exog_funcs = process_exog_funcs(options)

    # {'age': 0, 'married': 1, 'lagged_choice': 2, 'lagged_ltc': 3,
    # 'lagged_job_offer': 4, 'choice': 5}

    # [-2]: global exog state
    state_choice_vec = jnp.array([0, 0, 0, 2, 1])

    trans_vec = get_exog_transition_vec(
        state_choice_vec, exog_mapping, exog_funcs=exog_funcs, params=params
    )

    n_exog_states = sum(map(len, options["state_space"]["exogenous_states"].values()))
    assert np.equal(len(trans_vec), n_exog_states)

    # Call again since internal dictionary changed by function
    exog_funcs = process_exog_funcs(options)
    compute_exog_transition_vec = partial(
        get_exog_transition_vec, exog_mapping=exog_mapping, exog_funcs=exog_funcs
    )

    trans_vec_from_partial = compute_exog_transition_vec(
        state_choice_vec=state_choice_vec, params=params
    )
    aaae(trans_vec_from_partial, trans_vec)
