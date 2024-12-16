import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap

from dcegm.pre_processing.check_options import check_options_and_set_defaults
from dcegm.pre_processing.check_params import process_params
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.setup_model import (
    load_and_setup_model,
    setup_and_save_model,
    setup_model,
)
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options
from toy_models.cons_ret_model_dcegm_paper.budget_constraint import budget_constraint
from toy_models.cons_ret_model_dcegm_paper.state_space_objects import (
    create_state_space_function_dict,
)
from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)
from toy_models.cons_ret_model_dcegm_paper.utility_functions_log_crra import (
    utiility_log_crra,
)


def get_next_experience(period, lagged_choice, experience, options, params):

    working_hours = _transform_lagged_choice_to_working_hours(lagged_choice)

    return 1 / (period + 1) * (period * experience + (working_hours) / 3000)


def _transform_lagged_choice_to_working_hours(lagged_choice):

    not_working = lagged_choice == 0
    part_time = lagged_choice == 1
    full_time = lagged_choice == 2

    return not_working * 0 + part_time * 2000 + full_time * 3000


def util_wrap(state_dict, params, util_func):
    return util_func(**state_dict, params=params)


def test_wrap_function(load_replication_params_and_specs):
    params, _raw_options = load_replication_params_and_specs("deaton")
    options = {}

    options["model_params"] = _raw_options
    options.update(
        {
            "state_space": {
                "n_periods": 25,
                "choices": np.arange(2),
                "endogenous_states": {
                    "thus": np.arange(25),
                    "that": [0, 1],
                },
                "continuous_states": {
                    "wealth": np.linspace(0, 500, 100),
                },
                "exogenous_processes": {
                    "ltc": {"states": np.array([0]), "transition": jnp.array([0])}
                },
            },
        }
    )

    state_dict = {
        "consumption": jnp.arange(1, 7),
        "choice": jnp.arange(1, 7),
        "global_exog": np.zeros(6, dtype=int),
        "periods": jnp.arange(6),
    }

    util_expec = jnp.array(
        [0.0, 0.69314718, 1.09861229, 1.38629436, 1.60943791, 1.79175947], dtype=float
    )

    util_processed = determine_function_arguments_and_partial_options(
        utiility_log_crra,
        options,
    )

    calc_util = vmap(util_wrap, in_axes=(0, None, None))(
        state_dict, params, util_processed
    )
    assert jnp.allclose(calc_util, util_expec)


@pytest.mark.parametrize(
    "model_name",
    [
        ("retirement_no_taste_shocks"),
        ("retirement_taste_shocks"),
        ("deaton"),
    ],
)
def test_missing_parameter(
    model_name,
    load_replication_params_and_specs,
):
    params, _ = load_replication_params_and_specs(f"{model_name}")

    params.pop("interest_rate")
    params.pop("sigma")
    params.pop("lambda")

    params_dict = process_params(params)

    for param in ["interest_rate", "sigma", "lambda"]:
        assert param in params_dict.keys()

    params.pop("beta")
    with pytest.raises(ValueError, match="beta must be provided in params."):
        process_params(params)


@pytest.mark.parametrize(
    "model_name",
    [
        ("retirement_no_taste_shocks"),
        ("retirement_taste_shocks"),
        ("deaton"),
    ],
)
def test_load_and_save_model(
    model_name,
    load_replication_params_and_specs,
):
    options = {}
    _params, _raw_options = load_replication_params_and_specs(f"{model_name}")

    options["model_params"] = _raw_options
    options["model_params"]["n_choices"] = _raw_options["n_discrete_choices"]
    options["state_space"] = {
        "n_periods": 25,
        "choices": [i for i in range(_raw_options["n_discrete_choices"])],
        "continuous_states": {
            "wealth": np.linspace(0, 500, 100),
        },
    }
    options["exog_grids"] = options["state_space"]["continuous_states"]["wealth"].copy()

    model_setup = setup_model(
        options=options,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_constraint,
    )

    model_after_saving = setup_and_save_model(
        options=options,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_constraint,
        path="model.pkl",
    )

    model_after_loading = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_constraint,
        path="model.pkl",
    )

    for key in model_setup.keys():
        if isinstance(model_setup[key], np.ndarray):
            np.testing.assert_allclose(model_setup[key], model_after_loading[key])
            np.testing.assert_allclose(model_setup[key], model_after_saving[key])
        elif isinstance(model_setup[key], dict):
            for k in model_setup[key].keys():
                if isinstance(model_setup[key][k], np.ndarray):
                    np.testing.assert_allclose(
                        model_setup[key][k], model_after_loading[key][k]
                    )
                    np.testing.assert_allclose(
                        model_setup[key][k], model_after_saving[key][k]
                    )
                else:
                    pass
        else:
            pass

    import os

    os.remove("model.pkl")


def test_grid_parameters():
    options = {
        "model_params": {
            "saving_rate": 0.04,
        },
        "state_space": {
            "n_periods": 25,
            "choices": [0, 1],
            "continuous_states": {
                "wealth": np.linspace(0, 10, 100),
            },
        },
        "tuning_params": {
            "extra_wealth_grid_factor": 0.2,
            "n_constrained_points_to_add": 100,
        },
    }

    with pytest.raises(ValueError) as e:
        setup_model(
            options=options,
            state_space_functions=create_state_space_function_dict(),
            utility_functions=create_utility_function_dict(),
            utility_functions_final_period=create_final_period_utility_function_dict(),
            budget_constraint=budget_constraint,
        )


_periods = [0, 1, 2, 3, 4, 5]
_lagged_choices = [0, 1, 3]
_continuous_states = np.linspace(0, 1, 10)
TEST_INPUT = [
    (p, l, c) for p in _periods for l in _lagged_choices for c in _continuous_states
]


@pytest.mark.parametrize("period, lagged_choice, continuous_state", TEST_INPUT)
def test_second_continuous_state(period, lagged_choice, continuous_state):

    options = {
        "state_space": {
            "n_periods": 25,
            "choices": np.arange(3),
            # discrete states
            "endogenous_states": {
                "married": [0, 1],
                "n_children": np.arange(3),
            },
            "continuous_states": {
                "wealth": np.linspace(0, 10_000, 100),
                "experience": np.linspace(0, 1, 6),
            },
        },
        "model_params": {"savings_rate": 0.04},
    }
    params = {}

    state_space_functions = create_state_space_function_dict()
    state_space_functions["next_period_experience"] = get_next_experience

    options = check_options_and_set_defaults(options)

    model_funcs = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_constraint,
    )

    next_period_continuous_state = model_funcs["next_period_continuous_state"]

    got = next_period_continuous_state(
        period=period,
        lagged_choice=lagged_choice,
        continuous_state=continuous_state,
        options=options,
        params=params,
    )
    expected = get_next_experience(
        period=period,
        lagged_choice=lagged_choice,
        experience=continuous_state,
        options=options,
        params=params,
    )

    np.testing.assert_allclose(got, expected)
