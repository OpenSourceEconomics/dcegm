import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

import dcegm

SHOW_DEBUG_PLOTS = False


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
    return {
        "exp_red": exp_red + (lagged_choice == 0),
        "exp_green": exp_green + (lagged_choice == 1),
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
        # elif (exp_green > 0) & (lagged_choice == 0) & (period == 1):
        #    return False
        # elif (exp_red > 0) & (lagged_choice == 1) & (period == 1):
        #    return False
        # elif ((exp_red + exp_green) == 0) & (lagged_choice != 2):
        #    return False
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
        "assets_begin_of_period": jnp.ones(n_agents),
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
            "exp_green": jnp.arange(0, 7, dtype=float),
            "exp_red": jnp.arange(0, 7, dtype=float),
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

    # Also exercise ND interface path for DJ + additional continuous states.
    states_eval_cont = {
        "period": jnp.array([0, 1, 2], dtype=int),
        "lagged_choice": jnp.array([2, 0, 1], dtype=int),
        "exp_green": jnp.array([0.0, 1.0, 1.0]),
        "exp_red": jnp.array([0.0, 0.0, 1.0]),
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
        "exp_green": aligned_states_discrete["exp_green"].astype(float),
        "exp_red": aligned_states_discrete["exp_red"].astype(float),
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

    assert_allclose(
        float(jnp.mean(policy_gap[finite_policy])), 0.0, atol=1e-10, rtol=0.0
    )
    assert_allclose(float(jnp.mean(value_gap[finite_value])), 0.0, atol=2e-2, rtol=0.0)

    n_agents = 100
    states_initial = {
        "n_agents": n_agents,
        "assets_begin_of_period": jnp.ones(n_agents),
        "exp_green": jnp.zeros(n_agents),
        "exp_red": jnp.zeros(n_agents),
        "lagged_choice": jnp.zeros(n_agents),
        "period": jnp.zeros(n_agents, dtype=int),
    }
    simulate_cont_exp = model_cont_exp.get_solve_and_simulate_func(
        states_initial=states_initial, seed=99
    )

    df_cont_exp = simulate_cont_exp(params)

    # Compare simulated behavior between discrete-experience and continuous-experience
    # models under the same seed and initial states.

    choice_shares_discrete = (
        df.groupby("period").choice.value_counts(normalize=True).unstack(fill_value=0.0)
    )
    choice_shares_cont = (
        df_cont_exp.groupby("period")
        .choice.value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    all_choices_discrete_cont = sorted(
        set(choice_shares_discrete.columns).union(set(choice_shares_cont.columns))
    )
    choice_shares_discrete_aligned = choice_shares_discrete.reindex(
        columns=all_choices_discrete_cont,
        fill_value=0.0,
    )
    choice_shares_cont_aligned = choice_shares_cont.reindex(
        columns=all_choices_discrete_cont,
        fill_value=0.0,
    )
    choice_share_gap_discrete_cont = (
        choice_shares_discrete_aligned - choice_shares_cont_aligned
    ).abs()
    assert_allclose(
        float(choice_share_gap_discrete_cont.to_numpy().mean()),
        0.0,
        atol=0.2,
        rtol=0.0,
    )
    assert_allclose(
        float(choice_share_gap_discrete_cont.to_numpy().max()),
        0.0,
        atol=0.3,
        rtol=0.0,
    )

    consumption_means_discrete = df.groupby("period").consumption.mean()
    consumption_means_cont = df_cont_exp.groupby("period").consumption.mean()
    consumption_gap_discrete_cont = (
        consumption_means_discrete - consumption_means_cont
    ).abs()
    assert_allclose(
        float(consumption_gap_discrete_cont.mean()), 0.0, atol=0.7, rtol=0.0
    )
    assert_allclose(float(consumption_gap_discrete_cont.max()), 0.0, atol=1.0, rtol=0.0)

    # Test consumption by period and choice
    consumption_by_choice_discrete = (
        df.groupby(["period", "choice"]).consumption.mean().unstack(fill_value=0)
    )
    consumption_by_choice_cont = (
        df_cont_exp.groupby(["period", "choice"])
        .consumption.mean()
        .unstack(fill_value=0)
    )
    all_choices_cons = sorted(
        set(consumption_by_choice_discrete.columns).union(
            set(consumption_by_choice_cont.columns)
        )
    )
    consumption_by_choice_discrete_aligned = consumption_by_choice_discrete.reindex(
        columns=all_choices_cons, fill_value=0
    )
    consumption_by_choice_cont_aligned = consumption_by_choice_cont.reindex(
        columns=all_choices_cons, fill_value=0
    )
    consumption_by_choice_gap = (
        consumption_by_choice_discrete_aligned - consumption_by_choice_cont_aligned
    ).abs()
    assert_allclose(
        float(consumption_by_choice_gap.to_numpy().mean()), 0.0, atol=0.5, rtol=0.0
    )
    assert_allclose(
        float(consumption_by_choice_gap.to_numpy().max()), 0.0, atol=1.0, rtol=0.0
    )

    # Test exp_green distribution by period
    exp_green_dist_discrete = (
        df.groupby("period")
        .exp_green.value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    exp_green_dist_cont = (
        df_cont_exp.groupby("period")
        .exp_green.value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    all_exp_levels = sorted(
        set(exp_green_dist_discrete.columns).union(set(exp_green_dist_cont.columns))
    )
    exp_green_dist_discrete_aligned = exp_green_dist_discrete.reindex(
        columns=all_exp_levels, fill_value=0
    )
    exp_green_dist_cont_aligned = exp_green_dist_cont.reindex(
        columns=all_exp_levels, fill_value=0
    )
    exp_green_gap = (
        exp_green_dist_discrete_aligned - exp_green_dist_cont_aligned
    ).abs()
    assert_allclose(float(exp_green_gap.to_numpy().mean()), 0.0, atol=0.1, rtol=0.0)
    assert_allclose(float(exp_green_gap.to_numpy().max()), 0.0, atol=0.2, rtol=0.0)

    # Third model: continuous experience grid that does not align with integer years.
    model_config_cont_exp_offgrid = {
        "n_periods": 5,
        "choices": [0, 1, 2],
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(0, 50, 100),
            "assets_begin_of_period": jnp.linspace(0, 50, 100),
            "exp_green": jnp.arange(0.0, 6.0 + 1e-8, 1.8, dtype=float),
            "exp_red": jnp.arange(0.0, 6.0 + 1e-8, 1.8, dtype=float),
        },
        "n_quad_points": 5,
        "upper_envelope": {"method": "druedahl_jorgensen"},
    }

    model_cont_exp_offgrid = dcegm.setup_model(
        model_config=model_config_cont_exp_offgrid,
        model_specs=model_specs,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        state_space_functions=state_space_functions_cont_exp,
        stochastic_states_transitions={},
        budget_constraint=budget_constraint_cont_exp,
    )
    solved_model_cont_exp_offgrid = model_cont_exp_offgrid.solve(params)

    policy_cont_offgrid, value_cont_offgrid = (
        solved_model_cont_exp_offgrid.policy_and_value_for_states_and_choices(
            states=aligned_states_cont,
            choices=aligned_choices,
        )
    )

    finite_policy_offgrid = jnp.isfinite(policy_disc_aligned) & jnp.isfinite(
        policy_cont_offgrid
    )
    finite_value_offgrid = jnp.isfinite(value_disc_aligned) & jnp.isfinite(
        value_cont_offgrid
    )
    policy_gap_offgrid = jnp.abs(policy_disc_aligned - policy_cont_offgrid)
    value_gap_offgrid = jnp.abs(value_disc_aligned - value_cont_offgrid)

    # With a fine grid (0.2 step), queried integer-year points align exactly.
    assert_allclose(
        float(jnp.mean(policy_gap_offgrid[finite_policy_offgrid])),
        0.0,
        atol=1e-2,
        rtol=0.0,
    )
    assert_allclose(
        float(jnp.mean(value_gap_offgrid[finite_value_offgrid])),
        0.0,
        atol=1e-2,
        rtol=0.0,
    )

    # Expanded comparison between exact-grid and off-grid continuous models:
    # evaluate one query for each feasible discrete state-choice in every period,
    # while varying continuous states across queries.
    state_choice_space_cont = solved_model_cont_exp.model_structure[
        "state_choice_space"
    ]
    discrete_state_names_cont = solved_model_cont_exp.model_structure[
        "discrete_states_names"
    ]
    state_choice_cols = [*discrete_state_names_cont, "choice"]
    idx_period = state_choice_cols.index("period")
    idx_lagged_choice = state_choice_cols.index("lagged_choice")
    idx_choice = state_choice_cols.index("choice")

    periods_probe = state_choice_space_cont[:, idx_period].astype(int)
    lagged_probe = state_choice_space_cont[:, idx_lagged_choice].astype(int)
    choices_probe = state_choice_space_cont[:, idx_choice].astype(int)

    n_probe = periods_probe.shape[0]
    probe_idx = jnp.arange(n_probe)

    exp_green_probe = jnp.clip(
        0.35 * periods_probe
        + 0.11 * ((probe_idx % 5) - 2)
        + 0.07 * (lagged_probe == 1),
        0.0,
        6.0,
    )
    exp_red_probe = jnp.clip(
        0.45 * periods_probe
        + 0.09 * (((probe_idx * 3) % 7) - 3)
        + 0.05 * (lagged_probe == 0),
        0.0,
        6.0,
    )
    wealth_probe = jnp.clip(
        1.0 + 2.0 * periods_probe + 0.6 * (probe_idx % 9) + 0.25 * lagged_probe,
        0.5,
        49.5,
    )

    states_probe_cont = {
        "period": periods_probe,
        "lagged_choice": lagged_probe,
        "exp_green": exp_green_probe,
        "exp_red": exp_red_probe,
        "assets_begin_of_period": wealth_probe,
    }

    policy_cont_exact_probe, value_cont_exact_probe = (
        solved_model_cont_exp.policy_and_value_for_states_and_choices(
            states=states_probe_cont,
            choices=choices_probe,
        )
    )
    policy_cont_offgrid_probe, value_cont_offgrid_probe = (
        solved_model_cont_exp_offgrid.policy_and_value_for_states_and_choices(
            states=states_probe_cont,
            choices=choices_probe,
        )
    )

    finite_probe = (
        jnp.isfinite(policy_cont_exact_probe)
        & jnp.isfinite(value_cont_exact_probe)
        & jnp.isfinite(policy_cont_offgrid_probe)
        & jnp.isfinite(value_cont_offgrid_probe)
    )
    assert jnp.all(finite_probe)

    policy_probe_gap = jnp.abs(policy_cont_exact_probe - policy_cont_offgrid_probe)
    value_probe_gap = jnp.abs(value_cont_exact_probe - value_cont_offgrid_probe)

    assert_allclose(float(jnp.mean(policy_probe_gap)), 0.0, atol=1e-2, rtol=0.0)
    assert_allclose(float(jnp.max(policy_probe_gap)), 0.0, atol=2e-1, rtol=0.0)
    assert_allclose(float(jnp.mean(value_probe_gap)), 0.0, atol=5e-3, rtol=0.0)
    assert_allclose(float(jnp.max(value_probe_gap)), 0.0, atol=2e-2, rtol=0.0)

    simulate_cont_exp_offgrid = model_cont_exp_offgrid.get_solve_and_simulate_func(
        states_initial=states_initial,
        seed=99,
    )
    df_cont_exp_offgrid = simulate_cont_exp_offgrid(params)

    choice_shares_offgrid = (
        df_cont_exp_offgrid.groupby("period")
        .choice.value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )

    all_choices = sorted(
        set(choice_shares_discrete.columns).union(set(choice_shares_offgrid.columns))
    )
    choice_shares_discrete_aligned = choice_shares_discrete.reindex(
        columns=all_choices,
        fill_value=0.0,
    )

    choice_shares_offgrid_aligned = choice_shares_offgrid.reindex(
        columns=all_choices,
        fill_value=0.0,
    )

    choice_share_gap_offgrid = (
        choice_shares_discrete_aligned - choice_shares_offgrid_aligned
    ).abs()

    # Compare exact-grid and off-grid continuous models in simulation output.

    choice_shares_cont_aligned_for_offgrid = choice_shares_cont.reindex(
        columns=all_choices,
        fill_value=0.0,
    )
    choice_share_gap_cont_exact_offgrid = (
        choice_shares_cont_aligned_for_offgrid - choice_shares_offgrid_aligned
    ).abs()
    assert_allclose(
        float(choice_share_gap_cont_exact_offgrid.to_numpy().mean()),
        0.0,
        atol=1e-2,
        rtol=0.0,
    )
    assert_allclose(
        float(choice_share_gap_cont_exact_offgrid.to_numpy().max()),
        0.0,
        atol=1.01e-2,
        rtol=0.0,
    )

    consumption_means_offgrid = df_cont_exp_offgrid.groupby("period").consumption.mean()
    consumption_gap_cont_exact_offgrid = (
        consumption_means_cont - consumption_means_offgrid
    ).abs()
    assert_allclose(
        float(consumption_gap_cont_exact_offgrid.mean()),
        0.0,
        atol=2e-2,
        rtol=0.0,
    )
    assert_allclose(
        float(consumption_gap_cont_exact_offgrid.max()),
        0.0,
        atol=3e-2,
        rtol=0.0,
    )

    if SHOW_DEBUG_PLOTS:
        import matplotlib.pyplot as plt

        wealth_eval = jnp.linspace(0.5, 20.0, 300)
        choices_plot = jnp.zeros(wealth_eval.shape[0], dtype=int)

        states_plot_discrete = {
            "period": jnp.full(wealth_eval.shape[0], 3, dtype=int),
            "lagged_choice": jnp.zeros(wealth_eval.shape[0], dtype=int),
            "exp_green": jnp.ones(wealth_eval.shape[0], dtype=int),
            "exp_red": jnp.ones(wealth_eval.shape[0], dtype=int),
            "assets_begin_of_period": wealth_eval,
        }
        states_plot_cont = {
            "period": jnp.full(wealth_eval.shape[0], 3, dtype=int),
            "lagged_choice": jnp.zeros(wealth_eval.shape[0], dtype=int),
            "exp_green": jnp.ones(wealth_eval.shape[0], dtype=float),
            "exp_red": jnp.ones(wealth_eval.shape[0], dtype=float),
            "assets_begin_of_period": wealth_eval,
        }

        policy_discrete_plot, value_discrete_plot = (
            solved_model.policy_and_value_for_states_and_choices(
                states=states_plot_discrete,
                choices=choices_plot,
            )
        )
        policy_cont_exact_plot, value_cont_exact_plot = (
            solved_model_cont_exp.policy_and_value_for_states_and_choices(
                states=states_plot_cont,
                choices=choices_plot,
            )
        )
        policy_cont_offgrid_plot, value_cont_offgrid_plot = (
            solved_model_cont_exp_offgrid.policy_and_value_for_states_and_choices(
                states=states_plot_cont,
                choices=choices_plot,
            )
        )

        fig_policy, ax_policy = plt.subplots(figsize=(8, 4.5))
        ax_policy.plot(wealth_eval, policy_discrete_plot, label="discrete")
        ax_policy.plot(
            wealth_eval, policy_cont_exact_plot, label="continuous exact-grid"
        )
        ax_policy.plot(
            wealth_eval,
            policy_cont_offgrid_plot,
            label="continuous off-grid",
        )
        ax_policy.set_title("Policy By Wealth")
        ax_policy.set_xlabel("Assets at beginning of period")
        ax_policy.set_ylabel("Consumption")
        ax_policy.legend()

        fig_value, ax_value = plt.subplots(figsize=(8, 4.5))
        ax_value.plot(wealth_eval, value_discrete_plot, label="discrete")
        ax_value.plot(wealth_eval, value_cont_exact_plot, label="continuous exact-grid")
        ax_value.plot(
            wealth_eval,
            value_cont_offgrid_plot,
            label="continuous off-grid",
        )
        ax_value.set_title("Value By Wealth")
        ax_value.set_xlabel("Assets at beginning of period")
        ax_value.set_ylabel("Value")
        ax_value.legend()

        plt.show()

        choice_shares_offgrid.plot(
            kind="bar", stacked=True, title="Choice Shares - Off-grid Experience"
        )
        plt.show()
        choice_shares_discrete_aligned.plot(
            kind="bar", stacked=True, title="Choice Shares - Discrete Experience"
        )
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        df.groupby(["period", "choice"]).consumption.mean().unstack().fillna(0).plot(
            rot=0, title="Consumption - Discrete Experience Stocks", ax=ax[0]
        )

        df_cont_exp.groupby(["period", "choice"]).consumption.mean().unstack().fillna(
            0
        ).plot(rot=0, title="Consumption - Continuous Experience Stocks", ax=ax[1])

        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        df.groupby("period").exp_green.value_counts(normalize=True).unstack().plot(
            stacked=True,
            kind="bar",
            rot=0,
            title="Experience Green - Discrete Experience Stocks",
            ax=ax[0],
            cmap="Greens",
        )

        df_cont_exp.groupby("period").exp_green.value_counts(
            normalize=True
        ).unstack().plot(
            stacked=True,
            kind="bar",
            rot=0,
            title="Experience Green - Continuous Experience Stocks",
            ax=ax[1],
            cmap="Greens",
        )

        plt.show()
