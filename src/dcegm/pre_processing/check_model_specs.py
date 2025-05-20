import jax.numpy as jnp
import numpy as np


def extract_model_specs_info(model_specs):
    """Check if options are valid and set defaults."""
    # ToDo: Check these are all floats, ints or array single value arrays

    if not isinstance(model_specs, dict):
        raise ValueError("model_specs must be a dictionary.")

    # discount_factor processing
    if "discount_factor" in model_specs:
        read_func_discount_factor = lambda params: jnp.asarray(
            model_specs["discount_factor"]
        )
        discount_factor_in_params = False
    else:
        read_func_discount_factor = lambda params: jnp.asarray(
            params["discount_factor"]
        )
        discount_factor_in_params = True

    # interest_rate processing
    if "interest_rate" in model_specs:
        # Check if interest_rate is a scalar
        read_func_interest_rate = lambda params: jnp.asarray(
            model_specs["interest_rate"]
        )
        interest_rate_in_params = False
    else:
        read_func_interest_rate = lambda params: jnp.asarray(params["interest_rate"])
        interest_rate_in_params = True

    # income shock std processing ("income_shock_std")
    if "income_shock_std" in model_specs:
        # Check if income_shock_std is a scalar
        read_func_income_shock_std = lambda params: jnp.asarray(
            model_specs["income_shock_std"]
        )
        income_shock_std_in_params = False
    else:
        read_func_income_shock_std = lambda params: jnp.asarray(
            params["income_shock_std"]
        )
        income_shock_std_in_params = True

    # income shock std processing ("income_shock_std")
    if "income_shock_mean" in model_specs:
        # Check if income_shock_std is a scalar
        read_func_income_shock_mean = lambda params: jnp.asarray(
            model_specs["income_shock_mean"]
        )
        income_shock_mean_in_params = False
    else:
        read_func_income_shock_mean = lambda params: jnp.asarray(
            params["income_shock_mean"]
        )
        income_shock_mean_in_params = True

    specs_read_funcs = {
        "discount_factor": read_func_discount_factor,
        "interest_rate": read_func_interest_rate,
        "income_shock_std": read_func_income_shock_std,
        "income_shock_mean": read_func_income_shock_mean,
    }
    specs_params_info = {
        "discount_factor_in_params": discount_factor_in_params,
        "interest_rate_in_params": interest_rate_in_params,
        "income_shock_std_in_params": income_shock_std_in_params,
        "income_shock_mean_in_params": income_shock_mean_in_params,
    }

    return specs_read_funcs, specs_params_info
