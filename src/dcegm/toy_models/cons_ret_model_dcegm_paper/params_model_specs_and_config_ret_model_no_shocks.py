import jax.numpy as jnp


def example_params_retirement_no_shocks():

    params = {
        "discount_factor": 0.95,
        # disutility of work
        "delta": 0.35,
        # CRRA coefficient
        "rho": 1.95,
        # labor income coefficients
        "constant": 0.75,
        "exp": 0.04,
        "exp_squared": -0.0002,
        # Set shocks to zero (taste shock to almost zero. Not allowed to be zero)
        "income_shock_std": 0.00,
        "income_shock_mean": 0.0,
        "taste_shock_scale": 2.2204e-16,
        "interest_rate": 0.05,
        # consumption floor/retirement safety net (only relevant in the dc-egm retirement model)
        "consumption_floor": 0.001,
    }
    return params


def example_model_config_retirement_no_shocks():
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


def example_model_specs_retirement_no_shocks():
    model_specs = {
        "n_periods": 25,
        "min_age": 20,
        "n_choices": 2,
    }
    return model_specs
