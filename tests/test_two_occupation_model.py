"""Tests for the two-occupation retirement model.

The agent chooses between two occupations -- red (choice 0) and
green (choice 1) -- or retirement (choice 2, absorbing). Each
occupation accumulates its own experience stock (exp_red,
exp_green), which enters the wage equation.

Three model variants are compared: (1) discrete experience as
deterministic state variables, (2) continuous experience on an
integer-aligned grid, and (3) continuous experience on an
off-grid (step 1.8) that requires interpolation at integer
query points.

"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_allclose

import dcegm

jax.config.update("jax_enable_x64", True)

SHOW_DEBUG_PLOTS = False


# ====================================================================================
# Model functions
# ====================================================================================


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


def state_specific_choice_set(period, lagged_choice, model_specs):
    # Retirement is an absorbing state.
    if lagged_choice == 2:
        choice_set = [2]
    elif period == 4:
        choice_set = [2]
    else:
        choice_set = model_specs["choices"]
    return choice_set


def final_period_utility(wealth, choice, params):
    return flow_util(wealth, choice, params)


def marginal_final(wealth, choice, params):
    return marginal_utility(wealth, params)


def next_period_deterministic_state_cont(period, choice, lagged_choice):
    return {
        "period": period + 1,
        "lagged_choice": choice,
    }


def next_period_continuous_state(lagged_choice, period, exp_green, exp_red):
    return {
        "exp_red": exp_red + (lagged_choice == 0),
        "exp_green": exp_green + (lagged_choice == 1),
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


def next_period_deterministic_state_discrete(
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


def sparsity_condition(period, lagged_choice, exp_green, exp_red):
    if (exp_green + exp_red) > period:
        return False
    else:
        return True


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


# ====================================================================================
# Function dictionaries for model setup
# ====================================================================================

utility_functions = {
    "utility": flow_util,
    "inverse_marginal_utility": inverse_marginal_utility,
    "marginal_utility": marginal_utility,
}

utility_functions_final_period = {
    "utility": final_period_utility,
    "marginal_utility": marginal_final,
}

state_space_functions_cont_exp = {
    "state_specific_choice_set": state_specific_choice_set,
    "next_period_deterministic_state": next_period_deterministic_state_cont,
    "next_period_continuous_state": next_period_continuous_state,
}

state_space_functions_discrete_exp = {
    "state_specific_choice_set": state_specific_choice_set,
    "next_period_deterministic_state": next_period_deterministic_state_discrete,
    "sparsity_condition": sparsity_condition,
}


# ====================================================================================
# Assertion helpers
# ====================================================================================


def assert_alignment(
    solved_a, solved_b, states_a, states_b, choices, policy_atol, value_atol
):
    """Assert policy and value alignment between two solved models at given states."""
    policy_a, value_a = solved_a.policy_and_value_for_states_and_choices(
        states=states_a,
        choices=choices,
    )
    policy_b, value_b = solved_b.policy_and_value_for_states_and_choices(
        states=states_b,
        choices=choices,
    )
    finite_policy = jnp.isfinite(policy_a) & jnp.isfinite(policy_b)
    finite_value = jnp.isfinite(value_a) & jnp.isfinite(value_b)

    assert jnp.any(finite_policy)
    assert jnp.any(finite_value)

    policy_gap = jnp.abs(policy_a - policy_b)
    value_gap = jnp.abs(value_a - value_b)
    assert_allclose(
        float(jnp.mean(policy_gap[finite_policy])), 0.0, atol=policy_atol, rtol=0.0
    )
    assert_allclose(
        float(jnp.mean(value_gap[finite_value])), 0.0, atol=value_atol, rtol=0.0
    )


def assert_sim_shares_close(df_a, df_b, column, atol_mean, atol_max):
    """Assert distribution of column by period is close between two simulation DFs."""
    shares_a = (
        df_a.groupby("period")[column]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    shares_b = (
        df_b.groupby("period")[column]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    all_cols = sorted(set(shares_a.columns).union(set(shares_b.columns)))
    shares_a = shares_a.reindex(columns=all_cols, fill_value=0.0)
    shares_b = shares_b.reindex(columns=all_cols, fill_value=0.0)
    gap = (shares_a - shares_b).abs()
    assert_allclose(float(gap.to_numpy().mean()), 0.0, atol=atol_mean, rtol=0.0)
    assert_allclose(float(gap.to_numpy().max()), 0.0, atol=atol_max, rtol=0.0)


def assert_sim_means_close(df_a, df_b, column, group_by, atol_mean, atol_max):
    """Assert grouped means of column are close between two simulation DFs."""
    if isinstance(group_by, list) and len(group_by) > 1:
        means_a = df_a.groupby(group_by)[column].mean().unstack(fill_value=0)
        means_b = df_b.groupby(group_by)[column].mean().unstack(fill_value=0)
        all_cols = sorted(set(means_a.columns).union(set(means_b.columns)))
        means_a = means_a.reindex(columns=all_cols, fill_value=0)
        means_b = means_b.reindex(columns=all_cols, fill_value=0)
        gap = (means_a - means_b).abs()
        assert_allclose(float(gap.to_numpy().mean()), 0.0, atol=atol_mean, rtol=0.0)
        assert_allclose(float(gap.to_numpy().max()), 0.0, atol=atol_max, rtol=0.0)
    else:
        means_a = df_a.groupby(group_by)[column].mean()
        means_b = df_b.groupby(group_by)[column].mean()
        gap = (means_a - means_b).abs()
        assert_allclose(float(gap.mean()), 0.0, atol=atol_mean, rtol=0.0)
        assert_allclose(float(gap.max()), 0.0, atol=atol_max, rtol=0.0)


def show_debug_plots(
    solved_discrete,
    solved_cont_exp,
    solved_offgrid,
    df_discrete,
    df_cont_exp,
):
    """Visual comparison of policy, value, and choice shares."""

    wealth_eval = jnp.linspace(0.5, 20.0, 300)
    choices_plot = jnp.zeros(wealth_eval.shape[0], dtype=int)
    n = wealth_eval.shape[0]

    states_plot_discrete = {
        "period": jnp.full(n, 3, dtype=int),
        "lagged_choice": jnp.zeros(n, dtype=int),
        "exp_green": jnp.ones(n, dtype=int),
        "exp_red": jnp.ones(n, dtype=int),
        "assets_begin_of_period": wealth_eval,
    }
    states_plot_cont = {
        "period": jnp.full(n, 3, dtype=int),
        "lagged_choice": jnp.zeros(n, dtype=int),
        "exp_green": jnp.ones(n, dtype=float),
        "exp_red": jnp.ones(n, dtype=float),
        "assets_begin_of_period": wealth_eval,
    }

    policy_discrete_plot, value_discrete_plot = (
        solved_discrete.policy_and_value_for_states_and_choices(
            states=states_plot_discrete,
            choices=choices_plot,
        )
    )
    policy_cont_exact_plot, value_cont_exact_plot = (
        solved_cont_exp.policy_and_value_for_states_and_choices(
            states=states_plot_cont,
            choices=choices_plot,
        )
    )
    policy_cont_offgrid_plot, value_cont_offgrid_plot = (
        solved_offgrid.policy_and_value_for_states_and_choices(
            states=states_plot_cont,
            choices=choices_plot,
        )
    )

    fig_policy, ax_policy = plt.subplots(figsize=(8, 4.5))
    ax_policy.plot(wealth_eval, policy_discrete_plot, label="discrete")
    ax_policy.plot(wealth_eval, policy_cont_exact_plot, label="continuous exact-grid")
    ax_policy.plot(wealth_eval, policy_cont_offgrid_plot, label="continuous off-grid")
    ax_policy.set_title("Policy By Wealth")
    ax_policy.set_xlabel("Assets at beginning of period")
    ax_policy.set_ylabel("Consumption")
    ax_policy.legend()

    fig_value, ax_value = plt.subplots(figsize=(8, 4.5))
    ax_value.plot(wealth_eval, value_discrete_plot, label="discrete")
    ax_value.plot(wealth_eval, value_cont_exact_plot, label="continuous exact-grid")
    ax_value.plot(wealth_eval, value_cont_offgrid_plot, label="continuous off-grid")
    ax_value.set_title("Value By Wealth")
    ax_value.set_xlabel("Assets at beginning of period")
    ax_value.set_ylabel("Value")
    ax_value.legend()
    plt.show()

    choice_shares_discrete = (
        df_discrete.groupby("period")
        .choice.value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    choice_shares_discrete.plot(
        kind="bar", stacked=True, title="Choice Shares - Discrete Experience"
    )
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    df_discrete.groupby(["period", "choice"]).consumption.mean().unstack().fillna(
        0
    ).plot(rot=0, title="Consumption - Discrete Experience Stocks", ax=ax[0])
    df_cont_exp.groupby(["period", "choice"]).consumption.mean().unstack().fillna(
        0
    ).plot(rot=0, title="Consumption - Continuous Experience Stocks", ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    df_discrete.groupby("period").exp_green.value_counts(normalize=True).unstack().plot(
        stacked=True,
        kind="bar",
        rot=0,
        title="Experience Green - Discrete Experience Stocks",
        ax=ax[0],
        cmap="Greens",
    )
    df_cont_exp.groupby("period").exp_green.value_counts(normalize=True).unstack().plot(
        stacked=True,
        kind="bar",
        rot=0,
        title="Experience Green - Continuous Experience Stocks",
        ax=ax[1],
        cmap="Greens",
    )
    plt.show()


# ====================================================================================
# Fixtures
# ====================================================================================


@pytest.fixture(scope="module")
def params():
    return {
        "interest_rate": 0.02,
        "max_wealth": 50,
        "wage_constant": 3,
        "wage_exp_green": 0.5,
        "wage_exp_red": 0.8,
        "income_shock_std": 1,
        "income_shock_mean": 0,
        "taste_shock_scale": 1,
        "discount_factor": 0.95,
        "rho": 0.9,
        "delta": 1.5,
        "beta_green": 0.2,
        "beta_red": 0.1,
    }


@pytest.fixture(scope="module")
def model_specs():
    return {"choices": [0, 1, 2]}


@pytest.fixture(scope="module")
def discrete_model(model_specs):
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
    return dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        state_space_functions=state_space_functions_discrete_exp,
        stochastic_states_transitions={},
        budget_constraint=budget_constraint_discrete_exp,
    )


@pytest.fixture(scope="module")
def solved_discrete(discrete_model, params):
    return discrete_model.solve(params)


@pytest.fixture(scope="module")
def cont_exp_model(model_specs):
    model_config = {
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
    return dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        state_space_functions=state_space_functions_cont_exp,
        stochastic_states_transitions={},
        budget_constraint=budget_constraint_cont_exp,
    )


@pytest.fixture(scope="module")
def solved_cont_exp(cont_exp_model, params):
    return cont_exp_model.solve(params)


@pytest.fixture(scope="module")
def offgrid_model(model_specs):
    model_config = {
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
    return dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        state_space_functions=state_space_functions_cont_exp,
        stochastic_states_transitions={},
        budget_constraint=budget_constraint_cont_exp,
    )


@pytest.fixture(scope="module")
def solved_offgrid(offgrid_model, params):
    return offgrid_model.solve(params)


@pytest.fixture(scope="module")
def initial_states():
    n_agents = 100
    return {
        "n_agents": n_agents,
        "assets_begin_of_period": jnp.ones(n_agents),
        "exp_green": jnp.zeros(n_agents),
        "exp_red": jnp.zeros(n_agents),
        "lagged_choice": jnp.zeros(n_agents),
        "period": jnp.zeros(n_agents, dtype=int),
    }


@pytest.fixture(scope="module")
def df_discrete(discrete_model, initial_states, params):
    simulate = discrete_model.get_solve_and_simulate_func(
        states_initial=initial_states,
        seed=99,
    )
    return simulate(params)


@pytest.fixture(scope="module")
def df_cont_exp(cont_exp_model, initial_states, params):
    simulate = cont_exp_model.get_solve_and_simulate_func(
        states_initial=initial_states,
        seed=99,
    )
    return simulate(params)


@pytest.fixture(scope="module")
def df_offgrid(offgrid_model, initial_states, params):
    simulate = offgrid_model.get_solve_and_simulate_func(
        states_initial=initial_states,
        seed=99,
    )
    return simulate(params)


@pytest.fixture(scope="module")
def aligned_states():
    """Matched state points for comparing discrete vs continuous models."""
    discrete = {
        "period": jnp.array([2, 3, 3], dtype=int),
        "lagged_choice": jnp.array([0, 1, 0], dtype=int),
        "exp_green": jnp.array([0, 2, 1], dtype=int),
        "exp_red": jnp.array([1, 0, 1], dtype=int),
        "assets_begin_of_period": jnp.array([3.0, 6.0, 8.0]),
    }
    continuous = {
        "period": discrete["period"],
        "lagged_choice": discrete["lagged_choice"],
        "exp_green": discrete["exp_green"].astype(float),
        "exp_red": discrete["exp_red"].astype(float),
        "assets_begin_of_period": discrete["assets_begin_of_period"],
    }
    choices = jnp.array([0, 1, 0], dtype=int)
    return discrete, continuous, choices


# ====================================================================================
# Tests: discrete model interface
# ====================================================================================


def test_discrete_interface_joint_vs_separate(solved_discrete):
    """Joint policy+value query matches individual queries for the discrete model."""
    states_eval = {
        "period": jnp.array([0, 1, 2], dtype=int),
        "lagged_choice": jnp.array([2, 0, 1], dtype=int),
        "exp_green": jnp.array([0, 1, 1], dtype=int),
        "exp_red": jnp.array([0, 0, 1], dtype=int),
        "assets_begin_of_period": jnp.array([0.5, 4.0, 9.0]),
    }
    choices_eval = jnp.array([2, 0, 1], dtype=int)

    policy_joint, value_joint = solved_discrete.policy_and_value_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )
    policy_only = solved_discrete.policy_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )
    value_only = solved_discrete.value_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )

    assert jnp.allclose(policy_joint, policy_only, equal_nan=True)
    assert jnp.allclose(value_joint, value_only, equal_nan=True)


def test_discrete_interface_choice_values_match(solved_discrete):
    """Choice-wise value/policy arrays match direct state+choice queries."""
    states_eval = {
        "period": jnp.array([0, 1, 2], dtype=int),
        "lagged_choice": jnp.array([2, 0, 1], dtype=int),
        "exp_green": jnp.array([0, 1, 1], dtype=int),
        "exp_red": jnp.array([0, 0, 1], dtype=int),
        "assets_begin_of_period": jnp.array([0.5, 4.0, 9.0]),
    }
    choices_eval = jnp.array([2, 0, 1], dtype=int)

    value_only = solved_discrete.value_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )
    policy_only = solved_discrete.policy_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )

    choice_values_all = jnp.asarray(
        solved_discrete.choice_values_for_states(states=states_eval)
    )
    choice_policies_all = jnp.asarray(
        solved_discrete.choice_policies_for_states(states=states_eval)
    )
    idx = jnp.arange(choices_eval.shape[0])
    assert jnp.allclose(
        choice_values_all[idx, choices_eval], value_only, equal_nan=True
    )
    assert jnp.allclose(
        choice_policies_all[idx, choices_eval], policy_only, equal_nan=True
    )


def test_discrete_simulation_runs(df_discrete):
    """Discrete model simulation completes and returns a non-empty DataFrame."""
    assert len(df_discrete) > 0


# ====================================================================================
# Tests: continuous experience model interface
# ====================================================================================


def test_cont_exp_interface_joint_vs_separate(solved_cont_exp):
    """Joint policy+value query matches individual queries for the continuous model."""
    states_eval = {
        "period": jnp.array([0, 1, 2], dtype=int),
        "lagged_choice": jnp.array([2, 0, 1], dtype=int),
        "exp_green": jnp.array([0.0, 1.0, 1.0]),
        "exp_red": jnp.array([0.0, 0.0, 1.0]),
        "assets_begin_of_period": jnp.array([0.5, 4.0, 9.0]),
    }
    choices_eval = jnp.array([2, 0, 1], dtype=int)

    policy_joint, value_joint = solved_cont_exp.policy_and_value_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )
    policy_only = solved_cont_exp.policy_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )
    value_only = solved_cont_exp.value_for_states_and_choices(
        states=states_eval,
        choices=choices_eval,
    )

    assert jnp.allclose(policy_joint, policy_only, equal_nan=True)
    assert jnp.allclose(value_joint, value_only, equal_nan=True)


# ====================================================================================
# Tests: discrete vs continuous experience alignment
# ====================================================================================


def test_discrete_vs_cont_exp_alignment(
    solved_discrete,
    solved_cont_exp,
    aligned_states,
):
    """Policy/value match between discrete and continuous at integer states."""
    states_disc, states_cont, choices = aligned_states
    assert_alignment(
        solved_discrete,
        solved_cont_exp,
        states_disc,
        states_cont,
        choices,
        policy_atol=1e-10,
        value_atol=2e-2,
    )


def test_discrete_vs_cont_exp_simulation(df_discrete, df_cont_exp):
    """Simulated choice shares, consumption, and experience agree across models."""
    assert_sim_shares_close(
        df_discrete,
        df_cont_exp,
        "choice",
        atol_mean=0.2,
        atol_max=0.3,
    )
    assert_sim_means_close(
        df_discrete,
        df_cont_exp,
        "consumption",
        group_by="period",
        atol_mean=0.7,
        atol_max=1.0,
    )
    assert_sim_means_close(
        df_discrete,
        df_cont_exp,
        "consumption",
        group_by=["period", "choice"],
        atol_mean=0.5,
        atol_max=1.0,
    )
    assert_sim_shares_close(
        df_discrete,
        df_cont_exp,
        "exp_green",
        atol_mean=0.1,
        atol_max=0.2,
    )


# ====================================================================================
# Tests: off-grid continuous experience model
# ====================================================================================


def test_offgrid_vs_discrete_alignment(solved_discrete, solved_offgrid, aligned_states):
    """Off-grid model policy/value match discrete at integer states."""
    states_disc, states_cont, choices = aligned_states
    # With an off-grid (step 1.8), queried integer-year points require interpolation.
    assert_alignment(
        solved_discrete,
        solved_offgrid,
        states_disc,
        states_cont,
        choices,
        policy_atol=1e-2,
        value_atol=1e-2,
    )


def test_exact_vs_offgrid_probe(solved_cont_exp, solved_offgrid):
    """Exact-grid and off-grid models agree across state-choice probes."""
    state_choice_space = solved_cont_exp.model_structure["state_choice_space"]
    discrete_state_names = solved_cont_exp.model_structure["discrete_states_names"]
    state_choice_cols = [*discrete_state_names, "choice"]
    idx_period = state_choice_cols.index("period")
    idx_lagged_choice = state_choice_cols.index("lagged_choice")
    idx_choice = state_choice_cols.index("choice")

    periods_probe = state_choice_space[:, idx_period].astype(int)
    lagged_probe = state_choice_space[:, idx_lagged_choice].astype(int)
    choices_probe = state_choice_space[:, idx_choice].astype(int)

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

    states_probe = {
        "period": periods_probe,
        "lagged_choice": lagged_probe,
        "exp_green": exp_green_probe,
        "exp_red": exp_red_probe,
        "assets_begin_of_period": wealth_probe,
    }

    policy_exact, value_exact = solved_cont_exp.policy_and_value_for_states_and_choices(
        states=states_probe,
        choices=choices_probe,
    )
    policy_offgrid, value_offgrid = (
        solved_offgrid.policy_and_value_for_states_and_choices(
            states=states_probe,
            choices=choices_probe,
        )
    )

    finite_probe = (
        jnp.isfinite(policy_exact)
        & jnp.isfinite(value_exact)
        & jnp.isfinite(policy_offgrid)
        & jnp.isfinite(value_offgrid)
    )
    assert jnp.all(finite_probe)

    policy_gap = jnp.abs(policy_exact - policy_offgrid)
    value_gap = jnp.abs(value_exact - value_offgrid)

    assert_allclose(float(jnp.mean(policy_gap)), 0.0, atol=1e-2, rtol=0.0)
    assert_allclose(float(jnp.max(policy_gap)), 0.0, atol=2e-1, rtol=0.0)
    assert_allclose(float(jnp.mean(value_gap)), 0.0, atol=5e-3, rtol=0.0)
    assert_allclose(float(jnp.max(value_gap)), 0.0, atol=2e-2, rtol=0.0)


def test_exact_vs_offgrid_simulation(df_cont_exp, df_offgrid):
    """Simulated choice shares and consumption agree between exact and off-grid."""
    assert_sim_shares_close(
        df_cont_exp,
        df_offgrid,
        "choice",
        atol_mean=1e-2,
        atol_max=1.01e-2,
    )
    assert_sim_means_close(
        df_cont_exp,
        df_offgrid,
        "consumption",
        group_by="period",
        atol_mean=2e-2,
        atol_max=3e-2,
    )


def test_debug_plots(
    solved_discrete,
    solved_cont_exp,
    solved_offgrid,
    df_discrete,
    df_cont_exp,
):
    """Visual debug plots when SHOW_DEBUG_PLOTS is enabled."""
    if not SHOW_DEBUG_PLOTS:
        pytest.skip("SHOW_DEBUG_PLOTS is False")

    show_debug_plots(
        solved_discrete,
        solved_cont_exp,
        solved_offgrid,
        df_discrete,
        df_cont_exp,
    )
