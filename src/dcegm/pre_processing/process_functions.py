import functools
import inspect
from functools import partial


def determine_function_arguments_and_partial_options(
    func, options, additional_partial=None
):
    signature = set(inspect.signature(func).parameters)
    (
        partialed_func,
        signature,
    ) = partial_options_and_addtional_arguments_and_update_signature(
        func=func,
        signature=signature,
        options=options,
        additional_partial=additional_partial,
    )

    @functools.wraps(func)
    def processed_func(**kwargs):
        func_kwargs = {key: kwargs[key] for key in signature}

        return partialed_func(**func_kwargs)

    return processed_func


def partial_options_and_addtional_arguments_and_update_signature(
    func, signature, options, additional_partial
):
    """Partial in options and additional arguments and update signature."""
    if "options" in signature:
        func = partial(func, options=options)
        signature = signature - {"options"}
    if additional_partial is not None:
        func = partial(func, **additional_partial)
        signature = signature - set(additional_partial.keys())
    return func, signature
