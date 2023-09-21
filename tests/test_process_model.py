from pathlib import Path

import numpy as np
import pytest
from dcegm.process_model import _get_function_with_filtered_args_and_kwargs
from dcegm.process_model import convert_params_to_dict
from dcegm.process_model import process_exog_funcs
from dcegm.process_model import recursive_loop
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
    age,
    # married,
    lagged_ltc,
    # lagged_job_offer,
    *,
    ltc_prob_constant,
    ltc_prob_age,
    job_offer_constant,
    job_offer_age,
    job_offer_educ,
    job_offer_type_two,
):
    return (lagged_ltc == 0) * (ltc_prob_constant + age * ltc_prob_age) + (
        lagged_ltc == 1
    )


def func_exog_job_offer(
    age,
    married,
    # lagged_ltc,
    lagged_job_offer,
    *,
    ltc_prob_constant,
    ltc_prob_age,
    job_offer_constant,
    job_offer_age,
    job_offer_educ,
    job_offer_type_two,
):
    return (lagged_job_offer == 0) * job_offer_constant + (lagged_job_offer == 1) * (
        job_offer_constant + job_offer_type_two
    )


def func_exog_good_health(
    age,
    married,
    lagged_health,
    *,
    ltc_prob_constant,
    ltc_prob_age,
    job_offer_constant,
    job_offer_age,
    job_offer_educ,
    job_offer_type_two,
):
    return (
        (lagged_health == 0) * 0
        + (lagged_health == 1) * 0.3
        + (lagged_health == 2) * 0.7
    )


def func_exog_medium_health(
    age,
    married,
    lagged_health,
    *,
    ltc_prob_constant,
    ltc_prob_age,
    job_offer_constant,
    job_offer_age,
    job_offer_educ,
    job_offer_type_two,
):
    return (
        (lagged_health == 0) * 0
        + (lagged_health == 1) * 0.5
        + (lagged_health == 2) * 0.2
    )


def func_exog_bad_health(
    age,
    married,
    lagged_health,
    # choice,
    options,
    params,
    # *,
    # ltc_prob_constant,
    # ltc_prob_age,
    # job_offer_constant,
    # job_offer_age,
    # job_offer_educ,
    # job_offer_type_two,
):
    return (
        (lagged_health == 0) * 1
        + (lagged_health == 1) * 0.2
        + (lagged_health == 2) * 0.1
    )
    # return 0.7, 0.3, 0


def exog_health_mat(options):
    """Good, medium, bad health."""
    # mat = options["anytin"]

    # only allow to return vector, otherwise specify matrix
    return np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0, 0, 1]])


@pytest.fixture()
def example_exog_processes():
    """Define example exogenous processes."""

    # interface
    age = [0, 1]
    married = [0, 1]

    exog_ltc = [0, 1]
    exog_job_offer = [0, 1]

    # positional order matters
    state_vars = [age, married]
    exog_vars = [exog_ltc, exog_job_offer]

    params = {
        "ltc_prob_constant": 0.3,
        "ltc_prob_age": 0.1,
        "job_offer_constant": 0.5,
        "job_offer_age": 0,
        "job_offer_educ": 0,
        "job_offer_type_two": 0.4,
    }

    # mat = np.
    options = {
        # "model_structure":
        "exogenous_processes": {
            "ltc": [func_exog_ltc],
            "job_offer": np.array(),  # func_exog_job_offer,
        },
        "state_variables": {
            "endogenous": {
                "age": np.arange(2),
                "married": [0, 1],
                "lagged_choice": [0, 1],
            },
            "exogenous": {"lagged_ltc": [0, 1], "lagged_job_offer": [0, 1]},
        },
        # "model_params":
    }

    return state_vars, exog_vars, options, params


@pytest.fixture()
def test_data(load_example_model):
    n_grid = 10

    next_period_value = np.arange(n_grid)
    consumption = np.arange(n_grid) + 1

    params, _ = load_example_model("retirement_no_taste_shocks")
    params.loc[("utility_function", "theta"), "value"] = 1

    delta = params.loc[("delta", "delta"), "value"]
    beta = params.loc[("beta", "beta"), "value"]
    params = {"beta": beta, "delta": delta}

    compute_utility = utiility_func_log_crra

    return consumption, next_period_value, params, compute_utility


@pytest.mark.parametrize(
    "model",
    [
        ("retirement_no_taste_shocks"),
        ("retirement_taste_shocks"),
        ("deaton"),
    ],
)
def test_missing_beta(
    model,
    load_example_model,
):
    params, _ = load_example_model(f"{model}")
    params_without_beta = params.drop(index=("beta", "beta"))
    with pytest.raises(ValueError, match="Beta must be provided in params."):
        convert_params_to_dict(params_without_beta)


@pytest.mark.parametrize(
    "func",
    [
        utiility_func_log_crra,
        utility_func_crra,
        marginal_utility_crra,
        inverse_marginal_utility_crra,
    ],
)
def test_get_function_with_filtered_args_and_kwargs(func):
    state_vars_to_index = {
        "period": 0,
        "lagged_choice": 1,
        "married": 1,
        "exog_state": 2,
    }
    options = {"min_age": 50, "max_age": 80}

    func_with_filtered_args_and_kwargs = _get_function_with_filtered_args_and_kwargs(
        func, options=options, state_vars_to_index=state_vars_to_index
    )

    state_vec_full = np.array([10, 1, 9])
    kwargs = {
        "consumption": np.arange(10, 20),
        "choice": 0,
        "marginal_utility": np.arange(1, 11),
        "theta": 0.5,
        "delta": 0.1,
    }

    _util = func_with_filtered_args_and_kwargs(*state_vec_full, **kwargs)


def test_recursive_loop(example_exog_processes):
    state_vars, exog_vars, options, params = example_exog_processes

    # ============ state_vars ============

    _options = {"exogenous_processes": {"health": exog_health_mat()}}

    # func = _process_exog_funcs(_options, state_vars_to_index)

    # args = [0, 0, 0]
    # f = func[0][0]
    # out = func[0][0](*args, **params)

    # breakpoint()

    # ============ exog_funcs ============

    exog_funcs = process_exog_funcs(options)

    # Create a result array with the desired shape
    n_exog_states = np.prod([len(var) for var in exog_vars])
    result_shape = [n_exog_states, n_exog_states]
    result_shape.extend([len(var) for var in state_vars])

    transition_mat = np.zeros(result_shape)

    expected = np.array(
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

    recursive_loop(transition_mat, state_vars, exog_vars, [], [], exog_funcs, **params)

    aaae(transition_mat, expected)
