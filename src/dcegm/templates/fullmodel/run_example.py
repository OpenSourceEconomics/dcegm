# packages needed
import jax.numpy as jnp
import numpy as np
from jax import config

# import model_funcs
from model_funcs import (
    budget_constraint_exp,
    create_final_period_utility_function_dict,
    create_state_space_function_dict,
    create_utility_function_dict,
    job_offer,
    prob_survival,
)

import dcegm

config.update("jax_enable_x64", True)

## set up
n_periods = 20
n_choices = 3

model_config = {
    "min_period_batch_segments": [5, 12],
    "n_periods": n_periods,
    "choices": np.arange(n_choices, dtype=int),
    "deterministic_states": {
        "already_retired": np.arange(2, dtype=int),
    },
    "continuous_states": {
        "assets_end_of_period": jnp.arange(0, 100, 5, dtype=float),
        "experience": jnp.linspace(0, 1, 7, dtype=float),
    },
    "stochastic_states": {
        "job_offer": [0, 1],
        "survival": [0, 1],
    },
    "n_quad_points": 5,
}

stochastic_state_transitions = {
    "job_offer": job_offer,
    "survival": prob_survival,
}

model_specs = {
    "n_periods": n_periods,
    "n_choices": 3,
    "min_ret_period": 5,
    "max_ret_period": 10,
    "fresh_bonus": 0.1,
    "exp_scale": 20,
}

params = {
    "delta": 0.5,
    "discount_factor": 0.95,
    "taste_shock_scale": 1,
    "income_shock_std": 1,
    "interest_rate": 0.05,
    "constant": 1,
    "exp": 0.1,
    "exp_squared": -0.01,
    "consumption_floor": 0.5,
}

model = dcegm.setup_model(
    model_specs=model_specs,
    model_config=model_config,
    utility_functions=create_utility_function_dict(),
    utility_functions_final_period=create_final_period_utility_function_dict(),
    state_space_functions=create_state_space_function_dict(),
    budget_constraint=budget_constraint_exp,
    stochastic_states_transitions=stochastic_state_transitions,
)

# Solve the model
model_solved = model.solve(params)

# Simulate the model
n_agents = 1_000
states_initial = {
    "period": jnp.zeros(n_agents),
    "lagged_choice": jnp.ones(n_agents) * 2,  # all agents start as full time workers
    "already_retired": jnp.zeros(n_agents),
    "job_offer": jnp.ones(n_agents),
    "survival": jnp.ones(n_agents),
    "experience": jnp.ones(n_agents),
    "assets_begin_of_period": jnp.ones(n_agents) * 10,
}

model_solved.simulate(states_initial=states_initial, seed=42)
