import jax
import jax.numpy as jnp

import dcegm


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


def state_specific_choice_set(
    period,
    lagged_choice,
    model_specs,
):
    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if lagged_choice == 2:
        choice_set = [2]
    elif period == 4:
        choice_set = [2]
    else:
        choice_set = model_specs["choices"]

    return choice_set


def final_period_utility(wealth: float, choice: int, params):
    return flow_util(wealth, choice, params)


def marginal_final(wealth, choice, params):
    return marginal_utility(wealth, params)


utility_functions_final_period = {
    "utility": final_period_utility,
    "marginal_utility": marginal_final,
}


def next_period_deterministic_state_cont(
    period,
    choice,
    lagged_choice,
):
    return {
        "period": period + 1,
        "lagged_choice": choice,
    }


def next_period_continuous_state(
    lagged_choice,
    period,
    exp_green,
    exp_red,
):
    exp_red_years = period * exp_red
    exp_green_years = period * exp_green
    add_red = lagged_choice == 0
    add_green = lagged_choice == 1

    period_scale = period.clip(min=1)
    exp_red_lag_years = exp_red_years * (period_scale - 1).clip(min=0)
    exp_red_next = (exp_red_lag_years + add_red) / period_scale

    exp_green_lag_years = exp_green_years * (period_scale - 1).clip(min=0)
    exp_green_next = (exp_green_lag_years + add_green) / period_scale

    return {
        "exp_red": exp_red_next,
        "exp_green": exp_green_next,
    }


state_space_functions_cont_exp = {
    "state_specific_choice_set": state_specific_choice_set,
    "next_period_deterministic_state": next_period_deterministic_state_cont,
    "next_period_continuous_state": next_period_continuous_state,
}


def budget_constraint_cont_exp(
    period,
    lagged_choice,
    exp_green,
    exp_red,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    exp_green_years = exp_green * period
    exp_red_years = exp_red * period
    interest_factor = 1 + params["interest_rate"]
    wage = (
        params["wage_constant"]
        + params["wage_exp_green"] * exp_green_years * (lagged_choice == 1)
        + params["wage_exp_red"] * exp_red_years * (lagged_choice == 0)
    )
    resource = (
        interest_factor * asset_end_of_previous_period
        + (wage + income_shock_previous_period) * (lagged_choice != 2)
        + (wage + income_shock_previous_period) * 0.5 * (lagged_choice == 2)
    )
    return jnp.maximum(resource, 0.5)


def test_two_occupation_model_notebook_runs():
    jax.config.update("jax_enable_x64", True)

    params = {}
    params["interest_rate"] = 0.02
    params["max_wealth"] = 50
    params["wage_constant"] = 3
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
            "exp_green": jnp.arange(0, 7, dtype=int),
            "exp_red": jnp.arange(0, 7, dtype=int),
        },
        "n_quad_points": 5,
        "upper_envelope": {"method": "druedahl_jorgensen"},
    }

    def next_period_deterministic_state(
        period,
        choice,
        lagged_choice,
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
        if (exp_green + exp_red) > period:
            return False
        elif (exp_green > 0) & (lagged_choice == 0) & (period == 1):
            return False
        elif (exp_red > 0) & (lagged_choice == 1) & (period == 1):
            return False
        elif ((exp_red + exp_green) == 0) & (lagged_choice != 2):
            return False
        else:
            return True

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
        interest_factor = 1 + params["interest_rate"]
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

    solved_model = model.solve(params)
    policy_function = solved_model.policy

    # Exercise interface interpolation pipelines (policy/value/joint + choice-wise).
    states_eval = {
        "period": jnp.array([0, 1, 2], dtype=int),
        "lagged_choice": jnp.array([2, 0, 1], dtype=int),
        "exp_green": jnp.array([0, 1, 1], dtype=int),
        "exp_red": jnp.array([0, 0, 1], dtype=int),
        "assets_begin_of_period": jnp.array([0.5, 4.0, 9.0]),
    }
    choices_eval = jnp.array([2, 0, 1], dtype=int)

    policy_joint, value_joint = solved_model.policy_and_value_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )
    policy_only = solved_model.policy_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )
    value_only = solved_model.value_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )

    assert jnp.allclose(policy_joint, policy_only, equal_nan=True)
    assert jnp.allclose(value_joint, value_only, equal_nan=True)

    choice_values_all = jnp.asarray(
        solved_model.choice_values_for_states(states=states_eval)
    )
    choice_policies_all = jnp.asarray(
        solved_model.choice_policies_for_states(states=states_eval)
    )
    idx = jnp.arange(choices_eval.shape[0])
    assert jnp.allclose(
        choice_values_all[idx, choices_eval], value_only, equal_nan=True
    )
    assert jnp.allclose(
        choice_policies_all[idx, choices_eval], policy_only, equal_nan=True
    )

    n_agents = 100
    states_initial = {
        "n_agents": n_agents,
        "assets_begin_of_period": jnp.zeros(n_agents),
        "exp_green": jnp.zeros(n_agents),
        "exp_red": jnp.zeros(n_agents),
        "lagged_choice": jnp.zeros(n_agents),
        "period": jnp.zeros(n_agents, dtype=int),
    }

    simulate = model.get_solve_and_simulate_func(states_initial=states_initial, seed=99)

    df = simulate(params)

    model_config_cont_exp = {
        "n_periods": 5,
        "choices": [0, 1, 2],
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(0, 50, 100),
            "assets_begin_of_period": jnp.linspace(0, 50, 100),
            "exp_green": jnp.linspace(0, 1, 5, dtype=float),
            "exp_red": jnp.linspace(0, 1, 5, dtype=float),
        },
        "n_quad_points": 5,
        "upper_envelope": {"method": "druedahl_jorgensen"},
    }

    model_cont_exp = dcegm.setup_model(
        model_config=model_config_cont_exp,
        model_specs=model_specs,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        state_space_functions=state_space_functions_cont_exp,
        stochastic_states_transitions={},
        budget_constraint=budget_constraint_cont_exp,
    )

    # Multi-dimensional regular-grid interpolation for this setup is not yet
    # implemented in the solver path.
    solved_model_cont_exp = model_cont_exp.solve(params)
    policy_function_cont_exp = solved_model_cont_exp.policy

    # Also exercise ND interface path for DJ + additional continuous states.
    states_eval_cont = {
        "period": jnp.array([0, 1, 2], dtype=int),
        "lagged_choice": jnp.array([2, 0, 1], dtype=int),
        "exp_green": jnp.array([0.0, 0.25, 0.5]),
        "exp_red": jnp.array([0.0, 0.25, 0.5]),
        "assets_begin_of_period": jnp.array([0.5, 4.0, 9.0]),
    }
    choices_eval_cont = jnp.array([2, 0, 1], dtype=int)

    policy_joint_cont, value_joint_cont = (
        solved_model_cont_exp.policy_and_value_for_states_and_choices(
            states=states_eval_cont,
            choices=choices_eval_cont,
        )
    )
    policy_only_cont = solved_model_cont_exp.policy_for_states_and_choices(
        states=states_eval_cont,
        choices=choices_eval_cont,
    )
    value_only_cont = solved_model_cont_exp.value_for_states_and_choices(
        states=states_eval_cont,
        choices=choices_eval_cont,
    )

    assert jnp.allclose(policy_joint_cont, policy_only_cont, equal_nan=True)
    assert jnp.allclose(value_joint_cont, value_only_cont, equal_nan=True)

    # Explicit alignment checks between discrete-experience and continuous-experience
    # models at matched state points (continuous experience = discrete years / period).
    aligned_states_discrete = {
        "period": jnp.array([2, 3, 3], dtype=int),
        "lagged_choice": jnp.array([0, 1, 0], dtype=int),
        "exp_green": jnp.array([0, 2, 1], dtype=int),
        "exp_red": jnp.array([1, 0, 1], dtype=int),
        "assets_begin_of_period": jnp.array([3.0, 6.0, 8.0]),
    }
    aligned_choices = jnp.array([0, 1, 0], dtype=int)

    aligned_states_cont = {
        "period": aligned_states_discrete["period"],
        "lagged_choice": aligned_states_discrete["lagged_choice"],
        "exp_green": aligned_states_discrete["exp_green"]
        / aligned_states_discrete["period"],
        "exp_red": aligned_states_discrete["exp_red"]
        / aligned_states_discrete["period"],
        "assets_begin_of_period": aligned_states_discrete["assets_begin_of_period"],
    }

    policy_disc_aligned, value_disc_aligned = (
        solved_model.policy_and_value_for_states_and_choices(
            states=aligned_states_discrete,
            choices=aligned_choices,
        )
    )
    policy_cont_aligned, value_cont_aligned = (
        solved_model_cont_exp.policy_and_value_for_states_and_choices(
            states=aligned_states_cont,
            choices=aligned_choices,
        )
    )

    finite_policy = jnp.isfinite(policy_disc_aligned) & jnp.isfinite(
        policy_cont_aligned
    )
    finite_value = jnp.isfinite(value_disc_aligned) & jnp.isfinite(value_cont_aligned)

    assert jnp.any(finite_policy)
    assert jnp.any(finite_value)

    policy_gap = jnp.abs(policy_disc_aligned - policy_cont_aligned)
    value_gap = jnp.abs(value_disc_aligned - value_cont_aligned)

    assert jnp.mean(policy_gap[finite_policy]) < 2.5
    assert jnp.mean(value_gap[finite_value]) < 7.5

    n_agents = 100
    states_initial = {
        "n_agents": n_agents,
        "assets_begin_of_period": jnp.zeros(n_agents),
        "exp_green": jnp.zeros(n_agents),
        "exp_red": jnp.zeros(n_agents),
        "lagged_choice": jnp.zeros(n_agents),
        "period": jnp.zeros(n_agents, dtype=int),
    }

    simulate_cont_exp = model_cont_exp.get_solve_and_simulate_func(
        states_initial=states_initial, seed=99
    )

    df_cont_exp = simulate_cont_exp(params)

    assert policy_function is not None
    assert not df.empty
    assert model_config_cont_exp is not None
    assert state_space_functions_cont_exp is not None
    assert budget_constraint_cont_exp is not None
