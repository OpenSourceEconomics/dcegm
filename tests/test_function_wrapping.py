import functools
import inspect
from functools import partial

import jax.numpy as jnp
import numpy as np
from dcegm.process_model import _convert_params_to_dict
from jax import vmap
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_func_log_crra,
)


def test_function_wrapping(load_example_model):
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

    params = _convert_params_to_dict(params)

    state_dict = {
        "consumption": jnp.arange(1, 7),
        "choice": jnp.arange(1, 7),
        "global_exog": np.zeros(6, dtype=int),
        "periods": jnp.arange(6),
    }

    util_expec = jnp.array(
        [0.0, 0.69314718, 1.09861229, 1.38629436, 1.60943791, 1.79175947], dtype=float
    )

    util_processed = simple_wrapping(
        utiility_func_log_crra, options, {"ltc": np.array([0])}
    )

    calc_util = vmap(util_wrap, in_axes=(0, None, None))(
        state_dict, params, util_processed
    )
    assert jnp.allclose(calc_util, util_expec)


def util_wrap(state_dict, params, util_func):
    return util_func(**state_dict, params=params)


def simple_wrapping(func, options, exog_state_space):
    signature = set(inspect.signature(func).parameters)

    exogenous_processes_names = set(
        options["state_space"]["exogenous_processes"].keys()
    )

    exogs_in_signature = list(signature.intersection(exogenous_processes_names))
    signature_kwargs_without_exog = list(signature.difference(exogs_in_signature))

    exog_mapping = create_exog_mapt(exogs_in_signature, exog_state_space)

    options_processed_func = partial_options(func, signature, options)

    @functools.wraps(func)
    def processed_func(**kwargs):
        exog_kwargs = exog_mapping(kwargs["global_exog"])

        other_kwargs = {key: kwargs[key] for key in signature_kwargs_without_exog}

        return options_processed_func(**exog_kwargs, **other_kwargs)

    return processed_func


def partial_options(func, signature, options):
    if "options" in signature:
        return partial(func, options=options)
    else:
        return func


def create_exog_mapt(exogs_in_signature, exog_state_space):
    if len(exogs_in_signature) > 0:

        def exog_mapping(x):
            return {
                exog: jnp.take(exog_state_space[exog], x) for exog in exogs_in_signature
            }

    else:

        def exog_mapping(x):
            return {}

    return exog_mapping
