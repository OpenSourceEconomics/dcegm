from typing import Dict, Union

import pandas as pd


def process_params(params, params_check_info) -> Dict[str, float]:
    """Transforms params DataFrame into a dictionary.

    Checks if given params contains beta, taste shock scale, interest rate
    and discount factor.

    Args:
        params (dict or tuple or pandas.Series or pandas.DataFrame): Model parameters
            Support tuple and list as well?

    Returns:
        dict: Dictionary of model parameters.

    """

    if params_check_info["taste_shock_scale_in_params"]:
        if "taste_shock_scale" not in params.keys():
            raise ValueError(
                "There was no taste_shock_scale per state function provided and taste_shock_scale was"
                "not an element of model_specs or params."
            )

    if params_check_info["beta_in_params"]:
        if "discount_factor" not in params:
            raise ValueError("beta must be provided in params.")

    if params_check_info["interest_rate_in_params"]:
        if "interest_rate" not in params:
            raise ValueError("interest_rate must be provided in params.")

    if "interest_rate" not in params:
        params["interest_rate"] = 0
    if "sigma" not in params:
        params["sigma"] = 0
    if "income_shock_mean" not in params:
        params["income_shock_mean"] = 0

    return params
