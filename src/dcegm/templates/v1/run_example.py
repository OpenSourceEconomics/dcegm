# packages needed
import jax.numpy as jnp

# import model_funcs
from model_funcs import (
    budget_constraint,
    create_final_period_utility_function_dict,
    create_state_space_function_dict,
    create_utility_function_dict,
)

import dcegm

## set up
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
model_specs = {
    "min_age": 20,
    "n_choices": 2,
}
model_functions = {
    "utility_functions": create_utility_function_dict(),
    "utility_functions_final_period": create_final_period_utility_function_dict(),
    "state_space_functions": create_state_space_function_dict(),
    "budget_constraint": budget_constraint,
}

# Set up the model
model = dcegm.setup_model(
    model_config=model_config,
    model_specs=model_specs,
    **model_functions,
)

# Solve the model
model_solved = model.solve(params)

# Simulate the model
model_solved.simulate(
    seed=42,
)
