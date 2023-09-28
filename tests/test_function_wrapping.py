import inspect
from functools import partial

import jax.numpy as jnp
import numpy as np
from dcegm.process_model import _convert_params_to_dict
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

    global_dict = {
        "consumption": jnp.arange(1, 7),
        "choice": jnp.arange(1, 7),
        # "periods": jnp.arange(6),
        "params": params,
    }

    util_expec = jnp.array(
        [0.0, 0.69314718, 1.09861229, 1.38629436, 1.60943791, 1.79175947], dtype=float
    )

    calc_util = utiility_func_log_crra(**global_dict)
    assert jnp.allclose(calc_util, util_expec)


def exog_map(global_exog, exog_states_space):
    exog_map = {}
    for exog in exog_states_space.keys():
        exog_map[exog] = exog_states_space[exog][global_exog]
    return exog_map


def simple_wrapping(func, options, exog_state_space):
    signature = set(inspect.signature(func).parameters)

    exogenous_processes_names = set(
        options["state_space"]["exogenous_processes"].keys()
    )

    exogs_in_signature = list(signature.intersection(exogenous_processes_names))
    signature_kwargs_without_exog = list(signature.difference(exogs_in_signature))

    if len(exogs_in_signature) > 0:

        def exog_mapping(x):
            return {exog: exog_state_space[exog][x] for exog in exogs_in_signature}

    else:

        def exog_mapping(x):
            return {}

    if options in signature:
        partial_func = partial(func, options=options)
    else:
        partial_func = func

    def processed_func(kwargs):
        exog_kwargs = exog_mapping(kwargs["global_exog"])

        other_kwargs = {key: kwargs[key] for key in signature_kwargs_without_exog}

        return partial_func(**exog_kwargs, **other_kwargs)

    return processed_func
