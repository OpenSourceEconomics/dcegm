import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap

import dcegm.toy_models as toy_models
from dcegm.pre_processing.check_options import check_model_config_and_process
from dcegm.pre_processing.check_params import process_params
from dcegm.pre_processing.model_functions.process_model_functions import (
    process_model_functions,
)
from dcegm.pre_processing.setup_model import (  # load_and_setup_model,; setup_and_save_model,
    create_model_dict,
)
from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)
from dcegm.toy_models.cons_ret_model_dcegm_paper import (
    budget_constraint,
    create_final_period_utility_function_dict,
    create_state_space_function_dict,
    create_utility_function_dict,
    utility_crra,
)


def get_next_experience(period, lagged_choice, experience, model_specs, params):

    working_hours = _transform_lagged_choice_to_working_hours(lagged_choice)

    return 1 / (period + 1) * (period * experience + (working_hours) / 3000)


def _transform_lagged_choice_to_working_hours(lagged_choice):

    not_working = lagged_choice == 0
    part_time = lagged_choice == 1
    full_time = lagged_choice == 2

    return not_working * 0 + part_time * 2000 + full_time * 3000


def util_wrap(state_dict, params, util_func):
    return util_func(**state_dict, params=params)


def test_wrap_function():
    params, model_specs, _ = toy_models.load_example_params_model_specs_and_config(
        "dcegm_paper_deaton"
    )
    options = {}

    model_config = {
        "n_periods": 25,
        "choices": np.arange(2),
        "deterministic_states": {
            "thus": np.arange(25),
            "that": [0, 1],
        },
        "continuous_states": {
            "assets_end_of_period": np.linspace(0, 500, 100),
        },
        "stochastic_states": {"ltc": jnp.array([0])},
    }

    state_dict = {
        "consumption": jnp.arange(1, 7),
        "choice": jnp.arange(1, 7),
        "child_states_to_integrate_stochastic": np.zeros(6, dtype=int),
        "periods": jnp.arange(6),
    }

    util_expec = jnp.array(
        [0.0, 0.69314718, 1.09861229, 1.38629436, 1.60943791, 1.79175947], dtype=float
    )

    util_processed = determine_function_arguments_and_partial_model_specs(
        utility_crra,
        options,
    )

    calc_util = vmap(util_wrap, in_axes=(0, None, None))(
        state_dict, params, util_processed
    )
    assert jnp.allclose(calc_util, util_expec)


@pytest.mark.parametrize(
    "model_name",
    [
        ("retirement_no_shocks"),
        ("retirement_with_shocks"),
        ("deaton"),
    ],
)
def test_missing_parameter(
    model_name,
):
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            f"dcegm_paper_{model_name}"
        )
    )

    params.pop("interest_rate")
    params.pop("sigma")

    params_dict = process_params(params, {"taste_shock_scale_in_params": False})

    for param in ["interest_rate", "sigma"]:
        assert param in params_dict.keys()

    params.pop("beta")
    with pytest.raises(ValueError, match="beta must be provided in params."):
        process_params(params, {"taste_shock_scale_in_params": False})


# @pytest.mark.parametrize(
#     "model_name",
#     [
#         ("retirement_no_shocks"),
#         ("retirement_with_shocks"),
#         ("deaton"),
#     ],
# )
# def test_load_and_save_model(
#     model_name,
# ):
#     _params, model_specs, model_config = (
#         toy_models.load_example_params_model_specs_and_config(
#             "dcegm_paper_" + model_name
#         )
#     )
#
#     model_setup = create_model_dict(
#         model_config=model_config,
#         model_specs=model_specs,
#         state_space_functions=create_state_space_function_dict(),
#         utility_functions=create_utility_function_dict(),
#         utility_functions_final_period=create_final_period_utility_function_dict(),
#         budget_constraint=budget_constraint,
#     )
#
#     model_after_saving = setup_and_save_model(
#         model_config=model_config,
#         model_specs=model_specs,
#         state_space_functions=create_state_space_function_dict(),
#         utility_functions=create_utility_function_dict(),
#         utility_functions_final_period=create_final_period_utility_function_dict(),
#         budget_constraint=budget_constraint,
#         path="model.pkl",
#     )
#
#     model_after_loading = load_and_setup_model(
#         model_config=model_config,
#         model_specs=model_specs,
#         utility_functions=create_utility_function_dict(),
#         utility_functions_final_period=create_final_period_utility_function_dict(),
#         budget_constraint=budget_constraint,
#         state_space_functions=create_state_space_function_dict(),
#         path="model.pkl",
#     )
#
#     for key in model_setup.keys():
#         if isinstance(model_setup[key], np.ndarray):
#             np.testing.assert_allclose(model_setup[key], model_after_loading[key])
#             np.testing.assert_allclose(model_setup[key], model_after_saving[key])
#         elif isinstance(model_setup[key], dict):
#             for k in model_setup[key].keys():
#                 if isinstance(model_setup[key][k], np.ndarray):
#                     np.testing.assert_allclose(
#                         model_setup[key][k], model_after_loading[key][k]
#                     )
#                     np.testing.assert_allclose(
#                         model_setup[key][k], model_after_saving[key][k]
#                     )
#                 else:
#                     pass
#         else:
#             pass
#
#     import os
#
#     os.remove("model.pkl")


def test_grid_parameters():
    model_specs = {
        "saving_rate": 0.04,
    }
    model_config = {
        "n_periods": 25,
        "choices": [0, 1],
        "continuous_states": {
            "assets_end_of_period": np.linspace(0, 10, 100),
        },
        "tuning_params": {
            "extra_wealth_grid_factor": 0.2,
            "n_constrained_points_to_add": 100,
        },
    }

    with pytest.raises(ValueError) as e:
        create_model_dict(
            model_config=model_config,
            model_specs=model_specs,
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

    model_config = {
        "n_periods": 25,
        "n_quad_points": 5,
        "choices": np.arange(3),
        # discrete states
        "deterministic_states": {
            "married": [0, 1],
            "n_children": np.arange(3),
        },
        "continuous_states": {
            "assets_end_of_period": np.linspace(0, 10_000, 100),
            "experience": np.linspace(0, 1, 6),
        },
    }
    model_specs = {"savings_rate": 0.04}
    params = {}

    state_space_functions = create_state_space_function_dict()
    state_space_functions["next_period_experience"] = get_next_experience

    model_config = check_model_config_and_process(model_config)

    model_funcs, _ = process_model_functions(
        model_config=model_config,
        model_specs=model_specs,
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
        model_config=model_config,
        params=params,
    )
    expected = get_next_experience(
        period=period,
        lagged_choice=lagged_choice,
        experience=continuous_state,
        model_specs=model_specs,
        params=params,
    )

    np.testing.assert_allclose(got, expected)
