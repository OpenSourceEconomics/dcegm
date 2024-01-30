import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.pre_processing.params import process_params
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options
from jax import vmap
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_log_crra,
)


def util_wrap(state_dict, params, util_func):
    return util_func(**state_dict, params=params)


def test_wrap_function(load_example_model):
    params, _raw_options = load_example_model("deaton")
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

    params.pop("interest_rate")
    params.pop("sigma")
    params.pop("lambda")

    params_dict = process_params(params)

    for param in ["interest_rate", "sigma", "lambda"]:
        assert param in params_dict.keys()

    params.pop("beta")
    with pytest.raises(ValueError, match="beta must be provided in params."):
        process_params(params)
