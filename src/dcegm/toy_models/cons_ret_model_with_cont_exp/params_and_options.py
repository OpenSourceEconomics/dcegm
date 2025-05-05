import jax.numpy as jnp
import numpy as np


def example_params():
    params = {
        "beta": 0.95,
        "delta": 0.35,
        "rho": 1.95,
        "interest_rate": 0.04,
        "taste_shock_scale": 1,  # taste shock (scale) parameter
        "sigma": 1,  # shock on labor income, standard deviation
        "constant": 0.75,
        "exp": 0.04,
        "exp_squared": -0.0002,
        "consumption_floor": 0.001,
    }
    return params


def example_options():
    n_periods = 5
    n_choices = 2
    max_init_experience = 1

    options = {
        "model_params": {
            "n_choices": n_choices,
            "n_quad_points_stochastic": 5,
            "n_periods": n_periods,
            "max_init_experience": max_init_experience,
        },
        "state_space": {
            "n_periods": n_periods,
            "choices": np.arange(n_choices),
            "continuous_states": {
                "wealth": jnp.linspace(0, 50, 100),
                "experience": jnp.linspace(0, 1, 5),
            },
        },
    }
    return options
