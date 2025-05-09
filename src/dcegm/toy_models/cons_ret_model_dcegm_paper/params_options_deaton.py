from jax import numpy as jnp


def example_params_deaton():
    params = {
        # discount factor
        "beta": 0.95,
        # disutility of work
        "delta": 0,
        # CRRA coefficient
        "rho": 1,
        # labor income coefficients
        "constant": 0.75,
        "exp": 0.04,
        "exp_squared": -0.0004,
        "sigma": 0.25,
        # taste shock (scale) parameter
        "taste_shock_scale": 2.2204e-16,
        "interest_rate": 0.05,
        # Consumption floor. Needs to be there but not used
        "consumption_floor": 0.0,
    }
    return params


def example_options_deaton():

    options = {
        "state_space": {
            "n_periods": 25,
            "choices": [0],
            "continuous_states": {
                "wealth": jnp.linspace(
                    0,
                    75,
                    100,
                )
            },
        },
        "model_params": {
            "n_periods": 25,
            "min_age": 20,
            "n_choices": 1,
            "n_quad_points_stochastic": 10,
        },
    }
    return options
