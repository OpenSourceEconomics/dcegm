from typing import Dict, Union

import pandas as pd


def process_params(params, params_check_info) -> Dict[str, float]:
    """Transforms params DataFrame into a dictionary.

    Checks if given params contains discount_factor, taste shock scale, interest rate
    and discount factor.

    Args:
        params (dict or tuple or pandas.Series or pandas.DataFrame): Model parameters
            Support tuple and list as well?

    Returns:
        dict: Dictionary of model parameters.

    """
    # ToDo: Check these are all floats, ints or array single value arrays

    if params_check_info["taste_shock_scale_in_params"]:
        if "taste_shock_scale" not in params.keys():
            raise ValueError(
                "There was no taste_shock_scale per state function provided and taste_shock_scale was"
                "not an element of model_specs or params."
            )

    if params_check_info["discount_factor_in_params"]:
        if "discount_factor" not in params.keys():
            raise ValueError(
                "discount_factor must be provided in model_specs or params."
            )
    else:
        if "discount_factor" in params.keys():
            raise ValueError(
                "discount_factor was provided in params and model_specs. Choose one."
            )

    if params_check_info["interest_rate_in_params"]:
        if "interest_rate" not in params.keys():
            raise ValueError("interest_rate must be provided in model_specs or params.")
    else:
        if "interest_rate" in params.keys():
            raise ValueError(
                "interest_rate was provided in params and model_specs. Choose one."
            )

    if params_check_info["income_shock_std_in_params"]:
        if "income_shock_std" not in params.keys():
            raise ValueError(
                "income_shock_std must be provided in model_specs or params."
            )
    else:
        if "income_shock_std" in params.keys():
            raise ValueError(
                "income_shock_std was provided in params and model_specs. Choose one."
            )

    if params_check_info["income_shock_mean_in_params"]:
        if "income_shock_mean" not in params.keys():
            raise ValueError(
                "income_shock_mean must be provided in model_specs or params."
            )
    else:
        if "income_shock_mean" in params.keys():
            raise ValueError(
                "income_shock_mean was provided in params and model_specs. Choose one."
            )

    return params
