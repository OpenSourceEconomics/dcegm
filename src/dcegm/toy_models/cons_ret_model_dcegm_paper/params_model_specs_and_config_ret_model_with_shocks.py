import jax.numpy as jnp


def example_params_ret_model_with_shocks():
    params = {
        "discount_factor": 0.9523809523809523,
        # disutility of work
        "delta": 0.35,
        # CRRA coefficient
        "rho": 1.95,
        # labor income coefficients
        "constant": 0.75,
        "exp": 0.04,
        "exp_squared": -0.0002,
        # Shock parameters of income
        "income_shock_std": 0.35,
        "income_shock_mean": 0.0,
        "taste_shock_scale": 0.2,
        "interest_rate": 0.05,
        "consumption_floor": 0.001,
    }
    return params


def example_model_config_ret_model_with_shocks():

    model_config = {
        "n_periods": 25,
        "choices": [0, 1],
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(
                0,
                50,
                500,
            )
        },
        "n_quad_points": 5,
    }
    return model_config


def example_model_specs_ret_model_with_shocks():
    model_specs = {
        "min_age": 20,
        "n_choices": 2,
    }
    return model_specs
