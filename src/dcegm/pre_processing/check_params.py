from typing import Dict, Union

import pandas as pd


def process_params(params: Union[dict, pd.Series, pd.DataFrame]) -> Dict[str, float]:
    """Transforms params DataFrame into a dictionary.

    Checks if given params contains beta, taste shock scale, interest rate
    and discount factor.

    Args:
        params (dict or tuple or pandas.Series or pandas.DataFrame): Model parameters
            Support tuple and list as well?

    Returns:
        dict: Dictionary of model parameters.

    """

    if "interest_rate" not in params:
        params["interest_rate"] = 0
    if "sigma" not in params:
        params["sigma"] = 0
    if "beta" not in params:
        raise ValueError("beta must be provided in params.")
    if "income_shock_mean" not in params:
        params["income_shock_mean"] = 0

    return params
