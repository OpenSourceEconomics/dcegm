import functools
import inspect
from functools import partial

import numpy as np
from jax import numpy as jnp


def determine_function_arguments_and_partial_model_specs(
    func, model_specs, not_allowed_state_choices=None, continuous_state_name=None
):
    signature = set(inspect.signature(func).parameters)
    not_allowed_state_choices = (
        [] if not_allowed_state_choices is None else not_allowed_state_choices
    )

    partialed_func, signature = partial_model_specs_and_update_signature(
        func=func,
        signature=signature,
        model_specs=model_specs,
    )
    if len(not_allowed_state_choices) > 0:
        for var in signature:
            if var in not_allowed_state_choices:
                raise ValueError(
                    f"{func.__name__}() has a not allowed input variable: {var}"
                )

    @functools.wraps(func)
    def processed_func(**kwargs):

        if continuous_state_name:
            kwargs[continuous_state_name] = kwargs.get("continuous_state")

        func_kwargs = {key: kwargs[key] for key in signature}

        return partialed_func(**func_kwargs)

    return processed_func


def partial_model_specs_and_update_signature(func, signature, model_specs):
    """Partial in model_specs and update signature."""
    if "model_specs" in signature:
        func = partial(func, model_specs=model_specs)
        signature = signature - {"model_specs"}

    return func, signature


def create_array_with_smallest_int_dtype(arr):
    """Return array with the smallest unsigned integer dtype."""
    if isinstance(arr, (np.ndarray, jnp.ndarray)) and np.issubdtype(
        arr.dtype, np.integer
    ):
        return arr.astype(get_smallest_int_type(arr.max()))

    return arr


def get_smallest_int_type(n_values):
    """Return the smallest unsigned integer type that can hold n_values."""
    uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]

    for dtype in uint_types:
        if np.iinfo(dtype).max >= n_values:
            return dtype
