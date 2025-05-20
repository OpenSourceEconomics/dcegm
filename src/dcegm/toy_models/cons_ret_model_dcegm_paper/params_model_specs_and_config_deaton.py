from jax import numpy as jnp


def example_params_deaton():
    params = {
        # discount factor
        "discount_factor": 0.95,
        # disutility of work
        "delta": 0,
        # CRRA coefficient
        "rho": 1,
        # labor income coefficients
        "constant": 0.75,
        "exp": 0.04,
        "exp_squared": -0.0004,
        "income_shock_std": 0.25,
        "income_shock_mean": 0.0,
        # taste shock (scale) parameter
        "taste_shock_scale": 2.2204e-16,
        "interest_rate": 0.05,
        # Consumption floor. Needs to be there but not used
        "consumption_floor": 0.0,
    }
    return params


def example_model_config_deaton():
    model_config = {
        "n_periods": 25,
        "choices": [0],
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(
                0,
                75,
                100,
            )
        },
        "n_quad_points": 10,
    }
    return model_config


def example_model_specs_deaton():

    model_specs = {
        "n_periods": 25,
        "min_age": 20,
        "n_choices": 1,
    }
    return model_specs
