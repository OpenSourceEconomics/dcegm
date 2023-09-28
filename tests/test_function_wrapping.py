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
                "endogenous_states": {
                    "period": np.arange(25),
                    "lagged_choice": [0, 1],
                },
                "exogenous_states": {"exog_state": [0]},
                "choice": [i for i in range(_raw_options["n_discrete_choices"])],
            },
        }
    )

    params = _convert_params_to_dict(params)

    utiility_func_log_crra(consumption=0, choice=0, params=params)
