import jax.numpy as jnp


def example_params_ret_model_with_shocks():
    params = {
        "beta": 0.9523809523809523,
        # disutility of work
        "delta": 0.35,
        # CRRA coefficient
        "rho": 1.95,
        # labor income coefficients
        "constant": 0.75,
        "exp": 0.04,
        "exp_squared": -0.0002,
        # Shock parameters of income
        "sigma": 0.35,
        "taste_shock_scale": 0.2,
        "interest_rate": 0.05,
        "consumption_floor": 0.001,
    }
    return params


def example_options_ret_model_with_shocks():
    options = {
        "state_space": {
            "n_periods": 25,
            "choices": [0, 1],
            "continuous_states": {
                "wealth": jnp.linspace(
                    0,
                    50,
                    500,
                )
            },
        },
        "model_params": {
            "n_periods": 25,
            "min_age": 20,
            "n_choices": 2,
            "n_quad_points_stochastic": 5,
        },
    }
    return options
