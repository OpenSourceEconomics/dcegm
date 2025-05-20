import jax.numpy as jnp
import numpy as np


def extract_model_specs_info(model_specs):
    """Check if options are valid and set defaults."""

    if not isinstance(model_specs, dict):
        raise ValueError("model_specs must be a dictionary.")

    if "beta" in model_specs:
        # Check if beta is a scalar
        beta_val = model_specs["beta"]
        if not isinstance(beta_val, float):
            raise ValueError(
                f"beta is not a scalar of type float. got {beta_val} of type {type(beta_val)}"
            )
        read_func_beta = lambda params: jnp.asarray([model_specs["beta"]])
        beta_in_params = False
    else:
        read_func_beta = lambda params: jnp.asarray([model_funcs["read_funcs"]["beta"]])
        beta_in_params = True

    specs_read_funcs = {"beta": read_func_beta}
    specs_params_info = {"beta_in_params": beta_in_params}

    return specs_read_funcs, specs_params_info
