import functools
import inspect
from functools import partial


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


def determine_function_arguments_and_partial_options_beginning_of_period(
    func, options, continuous_state=None
):
    signature = set(inspect.signature(func).parameters)

    partialed_func, signature = partial_options_and_update_signature(
        func=func,
        signature=signature,
        options=options,
    )

    @functools.wraps(func)
    def processed_func(**kwargs):

        if continuous_state:
            kwargs[continuous_state] = kwargs["continuous_state"]

        func_kwargs = {key: kwargs[key] for key in signature}

        return partialed_func(**func_kwargs)

    return processed_func


def partial_options_and_update_signature(func, signature, options):
    """Partial in options and update signature."""
    if "options" in signature:
        func = partial(func, options=options)
        signature = signature - {"options"}

    return func, signature
