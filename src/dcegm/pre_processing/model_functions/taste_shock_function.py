from jax import numpy as jnp

from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options


def process_shock_functions(shock_functions, options, continuous_state_name):
    shock_functions = {} if shock_functions is None else shock_functions
    if "taste_shock_scale_per_state" in shock_functions.keys():
        taste_shock_scale_per_state = get_taste_shock_function_for_state(
            draw_function_taste_shocks=shock_functions["taste_shock_scale_per_state"],
            options=options,
            continuous_state_name=continuous_state_name,
        )
        taste_shock_is_scalar = False
    else:
        if "lambda" in options["model_params"]:
            # Check if lambda is a scalar
            lambda_val = options["model_params"]["lambda"]
            if not isinstance(lambda_val, (int, float)):
                raise ValueError(
                    f"Lambda is not a scalar. If there is no draw function provided, "
                    f"lambda must be a scalar. Got {lambda_val}."
                )
            taste_shock_scale_per_state = lambda state_dict_vec, params: jnp.asarray(
                [options["model_params"]["lambda"]]
            )
        else:
            taste_shock_scale_per_state = lambda state_dict_vec, params: jnp.asarray(
                [params["lambda"]]
            )

        taste_shock_is_scalar = True

    taste_shock_function_processed = {
        "taste_shock_scale_per_state": taste_shock_scale_per_state,
        "taste_shock_is_scalar": taste_shock_is_scalar,
    }
    return taste_shock_function_processed


def get_taste_shock_function_for_state(
    draw_function_taste_shocks, options, continuous_state_name
):
    not_allowed_states = ["wealth"]
    if continuous_state_name is not None:
        not_allowed_states += [continuous_state_name]
    taste_shock_scale_per_state_function = (
        determine_function_arguments_and_partial_options(
            func=draw_function_taste_shocks,
            options=options["model_params"],
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
