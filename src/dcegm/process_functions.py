import functools
import inspect
from functools import partial


def determine_function_arguments_and_partial_options(func, options):
    signature = set(inspect.signature(func).parameters)
    options_processed_func = partial_options(func, signature, options)

    @functools.wraps(func)
    def processed_func(**kwargs):
        func_kwargs = {key: kwargs[key] for key in signature}

        return options_processed_func(**func_kwargs)

    return processed_func


def partial_options(func, signature, options):
    if "options" in signature:
        return partial(func, options=options)
    else:
        return func
