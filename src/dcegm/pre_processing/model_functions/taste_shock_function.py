from jax import numpy as jnp

from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)


def process_shock_functions(shock_functions, model_specs, continuous_state_name):
    taste_shock_function_processed = {}
    shock_functions = {} if shock_functions is None else shock_functions
    if "taste_shock_scale_per_state" in shock_functions.keys():
        taste_shock_scale_per_state = get_taste_shock_function_for_state(
            draw_function_taste_shocks=shock_functions["taste_shock_scale_per_state"],
            model_specs=model_specs,
            continuous_state_name=continuous_state_name,
        )
        taste_shock_function_processed["taste_shock_scale_per_state"] = (
            taste_shock_scale_per_state
        )
        taste_shock_scale_is_scalar = False
        taste_shock_scale_in_params = False
    else:
        if "taste_shock_scale" in model_specs:
            # Check if lambda is a scalar
            lambda_val = model_specs["taste_shock_scale"]
            if not isinstance(lambda_val, (int, float)):
                raise ValueError(
                    f"Lambda is not a scalar. If there is no draw function provided, "
                    f"lambda must be a scalar. Got {lambda_val}."
                )
            read_func = lambda params: jnp.asarray([model_specs["taste_shock_scale"]])
            taste_shock_scale_in_params = False
        else:
            read_func = lambda params: jnp.asarray([params["taste_shock_scale"]])

            taste_shock_scale_in_params = True

        taste_shock_function_processed["read_out_taste_shock_scale"] = read_func

        taste_shock_scale_is_scalar = True

    taste_shock_function_processed["taste_shock_scale_is_scalar"] = (
        taste_shock_scale_is_scalar
    )

    return taste_shock_function_processed, taste_shock_scale_in_params


def get_taste_shock_function_for_state(
    draw_function_taste_shocks, continuous_state_name, model_specs
):
    not_allowed_states = ["assets_begin_of_period", "choice"]
    if continuous_state_name is not None:
        not_allowed_states += [continuous_state_name]
    taste_shock_scale_per_state_function = (
        determine_function_arguments_and_partial_model_specs(
            func=draw_function_taste_shocks,
            model_specs=model_specs,
            not_allowed_state_choices=not_allowed_states,
            continuous_state_name=continuous_state_name,
        )
    )

    def vectorized_taste_shock_scale_per_state(state_dict_vec, params):
        return taste_shock_scale_per_state_function(params=params, **state_dict_vec)

    # def taste_shock_over_all_states_func(state_dict, params):
    #     return jax.vmap(vectorized_taste_shock_scale_per_state, in_axes=(0, None))(
    #         state_dict, params
    #     )

    return vectorized_taste_shock_scale_per_state
