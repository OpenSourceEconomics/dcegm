# Code extracted from two_occupation_model.ipynb through the first simulation block.

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import dcegm

jax.config.update("jax_enable_x64", True)

params = {}
params["interest_rate"] = 0.02
params["max_wealth"] = 50
params["wage_constant"] = 300
params["wage_exp_green"] = 0.5
params["wage_exp_red"] = 0.8
params["income_shock_std"] = 1
params["income_shock_mean"] = 0
params["taste_shock_scale"] = 1
params["discount_factor"] = 0.95
params["rho"] = 0.9
params["delta"] = 1.5
params["beta_green"] = 0.2
params["beta_red"] = 0.1
params


# Utility functions
def flow_util(consumption, choice, params):
    rho = params["rho"]
    beta_green = params["beta_green"]
    beta_red = params["beta_red"]
    disutility = beta_red * (choice == 0) + beta_green * (choice == 1)
    u = consumption ** (1 - rho) / (1 - rho) - disutility
    return u


def marginal_utility(consumption, params):
    rho = params["rho"]
    u_prime = consumption ** (-rho)
    return u_prime


def inverse_marginal_utility(marginal_utility, params):
    rho = params["rho"]
    return marginal_utility ** (-1 / rho)


utility_functions = {
    "utility": flow_util,
    "inverse_marginal_utility": inverse_marginal_utility,
    "marginal_utility": marginal_utility,
}

# Final period utility functions.


def final_period_utility(wealth: float, choice: int, params):
    return flow_util(wealth, choice, params)


def marginal_final(wealth, choice, params):
    return marginal_utility(wealth, params)


utility_functions_final_period = {
    "utility": final_period_utility,
    "marginal_utility": marginal_final,
}


# Define state specific choice set.
def state_specific_choice_set(
    period,
    lagged_choice,
    model_specs,
):
    """State specific choice set limits which choices are available to agent given the
    state."""
    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if lagged_choice == 2:
        choice_set = [2]
    elif period == 4:
        choice_set = [2]
    else:
        choice_set = model_specs["choices"]

    return choice_set


# Model specifications.
model_specs = {
    "choices": [0, 1, 2],
}

model_config = {
    "n_periods": 5,
    "choices": [0, 1, 2],
    "continuous_states": {
        "assets_end_of_period": jnp.linspace(0, 50, 100),
        "assets_begin_of_period": jnp.linspace(0, 50, 100),
    },
    "deterministic_states": {
        "exp_green": jnp.arange(0, 6, dtype=int),
        "exp_red": jnp.arange(0, 6, dtype=int),
    },
    "n_quad_points": 5,
    "upper_envelope": {"method": "druedahl_jorgensen"},
}


def next_period_deterministic_state(
    period,
    choice,
    exp_green,
    exp_red,
):
    next_exp_green = exp_green + (choice == 1)
    next_exp_red = exp_red + (choice == 0)
    return {
        "period": period + 1,
        "exp_green": next_exp_green,
        "exp_red": next_exp_red,
        "lagged_choice": choice,
    }


def sparsity_condition(
    period,
    lagged_choice,
    exp_green,
    exp_red,
):
    """Define sparsity condition to rule out state space points that are not feasible
    given the model structure."""
    # Experience cannot exceed the period
    if (exp_green + exp_red) > period:
        return False
    # In later periods, if retired, shouldn't have accumulated experience before retirement
    else:
        return True


# Define dict of state space functions to pass to setuo_model.
state_space_functions_discrete_exp = {
    "state_specific_choice_set": state_specific_choice_set,
    "next_period_deterministic_state": next_period_deterministic_state,
    "sparsity_condition": sparsity_condition,
}


def budget_constraint_discrete_exp(
    lagged_choice,
    exp_green,
    exp_red,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    """Budget constraint that determines the resource available to the agent in the next
    period given the state and choice in the previous period."""
    interest_factor = 1 + params["interest_rate"]
    # Wage depends on accumulated experience in each occupation, retirees/unemployed only get wage constant.
    wage = (
        params["wage_constant"]
        + params["wage_exp_green"] * exp_green * (lagged_choice == 1)
        + params["wage_exp_red"] * exp_red * (lagged_choice == 0)
    )
    resource = (
        interest_factor * asset_end_of_previous_period
        + (wage + income_shock_previous_period) * (lagged_choice != 2)
        + (wage + income_shock_previous_period) * 0.5 * (lagged_choice == 2)
    )
    return jnp.maximum(resource, 0.5)


model = dcegm.setup_model(
    model_config=model_config,
    model_specs=model_specs,
    utility_functions=utility_functions,
    utility_functions_final_period=utility_functions_final_period,
    state_space_functions=state_space_functions_discrete_exp,
    stochastic_states_transitions={},
    budget_constraint=budget_constraint_discrete_exp,
)

n_agents = 1000
states_initial = {
    "n_agents": n_agents,
    "assets_begin_of_period": jnp.ones(n_agents),
    "exp_green": jnp.zeros(n_agents),
    "exp_red": jnp.zeros(n_agents),
    "lagged_choice": jnp.zeros(n_agents),
    "period": jnp.zeros(n_agents, dtype=int),
}
simulate = model.get_solve_and_simulate_func(states_initial=states_initial, seed=99)
df = simulate(params)

simulate = model.get_solve_and_simulate_func(states_initial=states_initial, seed=99)
df = simulate(params)
