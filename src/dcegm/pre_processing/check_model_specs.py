import jax.numpy as jnp
import numpy as np


def extract_model_specs_info(model_specs):
    """Check if options are valid and set defaults."""

    if not isinstance(model_specs, dict):
        raise ValueError("model_specs must be a dictionary.")

    if "discount_factor" in model_specs:
        # Check if discount_factor is a scalar
        discount_factor_val = model_specs["discount_factor"]
        if not isinstance(discount_factor_val, float):
            raise ValueError(
                f"discount_factor is not a scalar of type float. got {discount_factor_val} of type {type(discount_factor_val)}"
            )
        read_func_discount_factor = lambda params: jnp.asarray(
            [model_specs["discount_factor"]]
        )
        discount_factor_in_params = False
    else:
        read_func_discount_factor = lambda params: jnp.asarray(
            [model_specs["read_funcs"]["discount_factor"]]
        )
        discount_factor_in_params = True

    # interest_rate processing
    if "interest_rate" in model_specs:
        # Check if interest_rate is a scalar
        interest_rate_val = model_specs["interest_rate"]
        if not isinstance(interest_rate_val, float):
            raise ValueError(
                f"interest_rate is not a scalar of type float. got {interest_rate_val} of type {type(interest_rate_val)}"
            )
        read_func_interest_rate = lambda params: jnp.asarray(
            [model_specs["interest_rate"]]
        )
    else:
        read_func_interest_rate = lambda params: jnp.asarray(
            [model_specs["read_funcs"]["interest_rate"]]
        )

    specs_read_funcs = {
        "discount_factor": read_func_discount_factor,
        "interest_rate": read_func_interest_rate,
    }
    specs_params_info = {
        "discount_factor_in_params": discount_factor_in_params,
        "interest_rate_in_params": False,
    }

    return specs_read_funcs, specs_params_info
