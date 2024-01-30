import functools
import inspect
from functools import partial


def determine_function_arguments_and_partial_options(func, options):
    signature = set(inspect.signature(func).parameters)
    (
        partialed_func,
        signature,
    ) = partial_options_and_addtional_arguments_and_update_signature(
        func=func,
        signature=signature,
        options=options,
    )

    @functools.wraps(func)
    def processed_func(**kwargs):
        func_kwargs = {key: kwargs[key] for key in signature}

        return partialed_func(**func_kwargs)

    return processed_func


def partial_options_and_addtional_arguments_and_update_signature(
    func, signature, options
):
    """Partial in options and update signature."""
    if "options" in signature:
        func = partial(func, options=options)
        signature = signature - {"options"}

    return func, signature
