import functools
import inspect
from functools import partial

import numpy as np
from jax import numpy as jnp


def determine_function_arguments_and_partial_options(
    func, options, continuous_state_name=None
):
    signature = set(inspect.signature(func).parameters)

    partialed_func, signature = partial_options_and_update_signature(
        func=func,
        signature=signature,
        options=options,
    )

    @functools.wraps(func)
    def processed_func(**kwargs):

        if continuous_state_name:
            kwargs[continuous_state_name] = kwargs.get("continuous_state")

        func_kwargs = {key: kwargs[key] for key in signature}

        return partialed_func(**func_kwargs)

    return processed_func


def partial_options_and_update_signature(func, signature, options):
    """Partial in options and update signature."""
    if "options" in signature:
        func = partial(func, options=options)
        signature = signature - {"options"}

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
