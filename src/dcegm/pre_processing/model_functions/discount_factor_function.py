from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)


def process_discount_factor_function(
    shock_functions, model_specs, additional_continuous_state_names
):
    """Process a user-supplied state-choice-dependent discount factor.

    If `shock_functions["discount_factor_per_state"]` is provided, it is wrapped
    the same way as every other user function (utility, budget constraint, ...),
    so it can be called with `params` and whichever state-choice variables it
    declares in its signature. Otherwise, discount_factor stays the default
    single scalar read from `model_specs` or `params` (see
    `dcegm.pre_processing.check_model_specs.extract_model_specs_info`).

    """
    shock_functions = {} if shock_functions is None else shock_functions

    if "discount_factor_per_state" not in shock_functions:
        return None

    not_allowed_states = ["assets_begin_of_period"]
    if additional_continuous_state_names is not None:
        not_allowed_states += additional_continuous_state_names

    discount_factor_per_state_func = (
        determine_function_arguments_and_partial_model_specs(
            func=shock_functions["discount_factor_per_state"],
            model_specs=model_specs,
            not_allowed_state_choices=not_allowed_states,
        )
    )

    def read_func_discount_factor(params, **state_choice_vec):
        return discount_factor_per_state_func(params=params, **state_choice_vec)

    return read_func_discount_factor
