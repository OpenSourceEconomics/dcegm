from typing import Dict
from typing import Union

import pandas as pd
from pybaum import get_registry
from pybaum import tree_flatten


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

    if isinstance(params, (pd.Series, pd.DataFrame)):
        params = _convert_params_to_dict(params)

    if "interest_rate" not in params:
        params["interest_rate"] = 0
    if "lambda" not in params:
        params["lambda"] = 0
    if "sigma" not in params:
        params["sigma"] = 0
    if "beta" not in params:
        raise ValueError("beta must be provided in params.")

    return params


def _convert_params_to_dict(params: Union[pd.Series, pd.DataFrame]):
    """Converts params to dictionary."""
    _registry = get_registry(
        types=[
            "dict",
            "pandas.Series",
            "pandas.DataFrame",
        ],
        include_defaults=False,
    )
    # {level: df.xs(level).to_dict('index') for level in df.index.levels[0]}
    _params, _treedef = tree_flatten(params, registry=_registry)
    values = [i for i in _params if isinstance(i, (int, float))]
    keys = _treedef.index.get_level_values(_treedef.index.names[-1]).tolist()
    params_dict = dict(zip(keys, values))

    return params_dict
