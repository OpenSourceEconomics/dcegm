from functools import partial
from pathlib import Path

import numpy as np
import pytest
from dcegm.process_model import _get_function_with_filtered_args_and_kwargs
from dcegm.process_model import convert_params_to_dict
from dcegm.process_model import create_exog_mapping
from dcegm.process_model import create_exog_transition_mat_recursively
from dcegm.process_model import get_exog_transition_vec
from dcegm.process_model import get_utils_exog_processes
from dcegm.process_model import process_exog_funcs
from dcegm.process_model import process_exog_funcs_new
from dcegm.process_model import process_model_functions
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period_scalar,
)
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
    choice,
    *,
    ltc_prob_constant,
    ltc_prob_age,
    job_offer_constant,
    job_offer_age,
    job_offer_educ,
    job_offer_type_two,
):
    prob_ltc = (lagged_ltc == 0) * (ltc_prob_constant + age * ltc_prob_age) + (
        lagged_ltc == 1
    )
    prob_no_ltc = 1 - prob_ltc

    return prob_no_ltc, prob_ltc


def func_exog_job_offer(
    age,
    married,
    # choice,
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
    prob_job_offer = (lagged_job_offer == 0) * job_offer_constant + (
        lagged_job_offer == 1
    ) * (job_offer_constant + job_offer_type_two)
    prob_no_job_offer = 1 - prob_job_offer

    return prob_no_job_offer, prob_job_offer


def func_exog_health(
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

    options = {
        # "model_structure": {
        "exogenous_processes": {
            "ltc": func_exog_ltc,
            "job_offer": func_exog_job_offer,
        },
        "state_variables": {
            "endogenous": {
                "age": np.arange(2),
                "married": [0, 1],
                "lagged_choice": [0, 1],
                # "choice": [0, 1],
            },
            "exogenous": {"lagged_ltc": [0, 1], "lagged_job_offer": [0, 1]},
            "choice": [33, 333],
        },
        # "model_params": {}
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


def test_process_model_functions():
    options = {
        # "model_structure": {
        "exogenous_processes": {
            "ltc": func_exog_ltc,
            "job_offer": func_exog_job_offer,
        },
        "state_variables": {
            "endogenous": {
                "age": np.arange(2),
                "married": [0, 1],
                "lagged_choice": [0, 1],
                # "choice": [0, 1],
            },
            "exogenous": {"lagged_ltc": [0, 1], "lagged_job_offer": [0, 1]},
            "choice": [33, 333],
        },
        "model_params": {},
    }

    utility_funcs = {
        "utility": utility_func_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
        "marginal_utility": marginal_utility_crra,
    }

    process_model_functions(
        options,
        # state_vars,
        user_utility_functions=utility_funcs,
        user_budget_constraint=budget_constraint,
        user_final_period_solution=solve_final_period_scalar,
    )


def test_process_utility_funcs():
    options = {
        # "model_structure": {
        "exogenous_processes": {
            "ltc": func_exog_ltc,
            "job_offer": func_exog_job_offer,
        },
        "state_variables": {
            "endogenous": {
                "period": np.arange(2),
                "age": np.arange(2),
                "married": [0, 1],
                "lagged_choice": [0, 1],
                # "choice": [0, 1],
            },
            "exogenous": {"lagged_ltc": [0, 1], "lagged_job_offer": [0, 1]},
            "choice": [33, 333],
        },
        "model_params": {},
    }

    params = {"theta": 0.3, "delta": 0.5, "beta": 0.9}

    state_vars_and_choice = (
        options["state_variables"]["endogenous"]
        | options["state_variables"]["exogenous"]
        | {"choice": options["state_variables"]["choice"]}
    )
    state_vars_and_choice_to_index = {
        key: idx for idx, key in enumerate(state_vars_and_choice)
    }

    compute_utility = _get_function_with_filtered_args_and_kwargs(
        utility_func_crra,
        options=options,
        state_vars_to_index=state_vars_and_choice_to_index,
    )

    policy = np.arange(10)

    # ['period', 'age, 'married', 'lagged_choice', 'lagged_ltc',
    # 'lagged_job_offer', 'choice']
    state_choice_vec = [0, 0, 0, 0, 0, 1, 10]

    compute_utility(consumption=policy, *state_choice_vec, **params)


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
    options = {"model_params": {"min_age": 50, "max_age": 80}}

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
        "options": options,
    }

    _util = func_with_filtered_args_and_kwargs(*state_vec_full, **kwargs)


@pytest.mark.skip
def test_recursive_loop(example_exog_processes):
    state_vars_and_choice, exog_vars, options, params = example_exog_processes

    # ============ state_vars ============

    # _options = {"exogenous_processes": {"health": exog_health_mat}}

    # func = _process_exog_funcs(_options, state_vars_to_index)

    # args = [0, 0, 0]
    # f = func[0][0]
    # out = func[0][0](*args, **params)

    # breakpoint()

    # ============ exog_funcs ============

    state_vars_and_choice = (
        options["state_variables"]["endogenous"]
        | options["state_variables"]["exogenous"]
        | {"choice": options["state_variables"]["choice"]}
    )
    state_vars_to_index = {
        key: idx for idx, key in enumerate(state_vars_and_choice.keys())
    }
    _, signature = process_exog_funcs(options, state_vars_to_index)

    exog_utils, exog_shape = get_utils_exog_processes(options)
    transition_mat = np.empty(exog_shape)

    create_exog_transition_mat_recursively(
        transition_mat=transition_mat,
        **exog_utils["recursion"] | params | params,
    )

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

    if "choice" in signature:
        aaae(transition_mat, np.stack([expected, expected], axis=4))
    else:
        aaae(transition_mat, expected)


def test_exog_mapping():
    options = {
        "state_variables": {
            "exogenous": {"exog_1": [1, 2, 3], "exog_2": [4, 5], "exog_3": [6, 7, 8, 9]}
        }
    }

    # =================================================================================

    exog_state_vars = options["state_variables"]["exogenous"]

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


def test_get_exog_transition_vec():
    options = {
        # "model_structure": {
        "exogenous_processes": {
            "ltc": func_exog_ltc,
            "job_offer": func_exog_job_offer,
        },
        "state_variables": {
            "endogenous": {
                "age": np.arange(2),
                "married": [0, 1],
                "lagged_choice": [0, 1],
                # "choice": [0, 1],
            },
            "exogenous": {"lagged_ltc": [0, 1], "lagged_job_offer": [0, 1]},
            "choice": [0, 1],
        },
        # "model_params": {}
    }

    params = {
        "ltc_prob_constant": 0.3,
        "ltc_prob_age": 0.1,
        "job_offer_constant": 0.5,
        "job_offer_age": 0,
        "job_offer_educ": 0,
        "job_offer_type_two": 0.4,
    }

    exog_mapping = create_exog_mapping(options)
    exog_funcs, _signature = process_exog_funcs_new(options)

    # {'age': 0, 'married': 1, 'lagged_choice': 2, 'lagged_ltc': 3,
    # 'lagged_job_offer': 4, 'choice': 5}

    # [-2]: global exog state
    state_choice_vec = np.array([0, 0, 0, 2, 1])

    trans_vec = get_exog_transition_vec(
        state_choice_vec, exog_mapping, exog_funcs=exog_funcs, params=params
    )

    n_exog_states = sum(map(len, options["state_variables"]["exogenous"].values()))
    assert np.equal(len(trans_vec), n_exog_states)

    compute_exog_transition_vec = partial(
        get_exog_transition_vec, exog_mapping=exog_mapping, exog_funcs=exog_funcs
    )

    trans_vec_from_partial = compute_exog_transition_vec(
        state_choice_vec=state_choice_vec, params=params
    )
    aaae(trans_vec_from_partial, trans_vec)
