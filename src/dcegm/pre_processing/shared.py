import functools
import inspect
from functools import partial


def determine_function_arguments_and_partial_options(
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

        # if continuous_state:
        #     if "continuous_state" in kwargs:
        #         kwargs[continuous_state] = kwargs["continuous_state"]
        #     elif "continuous_state_beginning_of_period" in kwargs:
        #         kwargs[continuous_state] = kwargs[
        #             "continuous_state_beginning_of_period"
        #         ]
        if continuous_state:
            kwargs[continuous_state] = kwargs.get(
                "continuous_state", kwargs.get("continuous_state_beginning_of_period")
            )

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

        # To-Do: Quick fix for now. Check if this is necessary
        if (
            "choice" in signature
            and "choice" not in kwargs
            and continuous_state in signature
            and "lagged_choice" in kwargs
        ):
            kwargs["choice"] = kwargs["lagged_choice"]
            # signature.add("lagged_choice")
            # signature.discard("choice")

            print("Replace")

        func_kwargs = {key: kwargs[key] for key in signature}

        return partialed_func(**func_kwargs)

    return processed_func


def partial_options_and_update_signature(func, signature, options):
    """Partial in options and update signature."""
    if "options" in signature:
        func = partial(func, options=options)
        signature = signature - {"options"}

    return func, signature
