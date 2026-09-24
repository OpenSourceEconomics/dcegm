"""Closed-form Euler-equation oracle for the death (value-reuse proxy) redirect.

A three-period toy in which the agent dies with certainty after period 0. The period-0
parent's only child is therefore a death state at period 1, which proxies across periods
to the single last-period (period 2) death slot -- the exact cross-period redirect the
decoupling has to get right.

Death is consume-all, so the child's marginal value is analytic (``bequest'(wealth,
experience)``), which makes the parent's Euler equation closed form despite the proxy.
The bequest depends on experience, so the check is sensitive to the child's own (non-
proxy) continuation being read correctly. If the solve reproduces this Euler equation,
the proxy value redirect plus the non-proxy transition are correct.

"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm

import dcegm

N_PERIODS = 3
LAST = N_PERIODS - 1

_PARAMS = {
    "discount_factor": 0.95,
    "interest_rate": 0.05,
    "income_shock_std": 0.1,
    "income_shock_mean": 0.0,
    "taste_shock_scale": 1.0,
    "wage": 1.0,
    "wage_growth": 0.5,
    "consumption_floor": 1e-8,
    "bequest_exp_slope": 0.8,
    "delta": 0.0,
}
_MODEL_CONFIG = {
    "n_periods": N_PERIODS,
    "choices": np.arange(1, dtype=int),
    "continuous_states": {
        "assets_end_of_period": jnp.linspace(0, 20, 12, dtype=float),
        "experience": jnp.linspace(0, 1, 3, dtype=float),
    },
    "stochastic_states": {"survival": [0, 1]},
    "n_quad_points": 5,
}


# =====================================================================================
# Model functions
# =====================================================================================


def _utility(consumption, survival, experience, choice, params):
    alive = jnp.log(consumption) - choice * params["delta"]
    dead = (1 + params["bequest_exp_slope"] * experience) * jnp.log(consumption)
    return jnp.where(survival == 0, dead, alive)


def _marginal_utility(consumption, survival, experience, choice, params):
    dead = (1 + params["bequest_exp_slope"] * experience) / consumption
    return jnp.where(survival == 0, dead, 1 / consumption)


def _inverse_marginal_utility(marginal_utility, params):
    # Only alive (non-terminal) states invert the Euler equation; death is consume-all.
    return 1 / marginal_utility


def _utility_final(wealth, survival, experience, params):
    alive = jnp.log(wealth)
    dead = (1 + params["bequest_exp_slope"] * experience) * jnp.log(wealth)
    return jnp.where(survival == 0, dead, alive)


def _marginal_utility_final(wealth, survival, experience, params):
    dead = (1 + params["bequest_exp_slope"] * experience) / wealth
    return jnp.where(survival == 0, dead, 1 / wealth)


def _budget(
    period, asset_end_of_previous_period, income_shock_previous_period, choice, params
):
    # Declares ``choice`` (numerically a no-op with a single choice) so the model
    # takes the choice-dependent law-of-motion branch -- i.e. so the death child is
    # read out of ``law_of_motion_arrays["child_state_choices"]`` rather than the
    # coarser per-state dedup, exercising the branch this oracle targets. Income
    # grows with ``period`` so the redirect is load-bearing: reading the death
    # child at its real age (period 1) versus the proxy's period (the last period)
    # would change the transition and break the Euler equation below.
    wage = params["wage"] * (1 + params["wage_growth"] * period)
    income = jnp.exp(income_shock_previous_period) * wage + 0.0 * choice
    wealth = (1 + params["interest_rate"]) * asset_end_of_previous_period + income
    return jnp.maximum(wealth, params["consumption_floor"])


def _next_period_experience(experience):
    # Identity: keeps the child's experience on a grid node, so the death value is
    # read at an exact node and the closed-form comparison is not clouded by
    # experience-axis interpolation.
    return experience


def _next_period_continuous_state(experience):
    return {"experience": _next_period_experience(experience)}


def _next_period_deterministic_state(period, choice):
    return {"period": period + 1, "lagged_choice": choice}


def _sparsity_condition(period, lagged_choice, survival):
    # Living states are solved in place; every death state redirects to the single
    # last-period death slot (the cross-period value-reuse proxy).
    if survival == 0:
        return {"period": LAST, "lagged_choice": lagged_choice, "survival": 0}
    return True


def _state_specific_choice_set(period, lagged_choice):
    return np.arange(1, dtype=int)


def _prob_survival():
    # Death next period with certainty, so the period-0 parent's only child is a
    # death state (proxied), which makes its Euler equation closed form.
    return jnp.array([1.0, 0.0])


def _setup_and_solve():
    model = dcegm.setup_model(
        model_specs={"n_periods": N_PERIODS},
        model_config=_MODEL_CONFIG,
        utility_functions={
            "utility": _utility,
            "marginal_utility": _marginal_utility,
            "inverse_marginal_utility": _inverse_marginal_utility,
        },
        utility_functions_final_period={
            "utility": _utility_final,
            "marginal_utility": _marginal_utility_final,
        },
        state_space_functions={
            "state_specific_choice_set": _state_specific_choice_set,
            "next_period_continuous_state": _next_period_continuous_state,
            "next_period_deterministic_state": _next_period_deterministic_state,
            "sparsity_condition": _sparsity_condition,
        },
        budget_constraint=_budget,
        stochastic_states_transitions={"survival": _prob_survival},
    )
    return model, model.solve(params=_PARAMS)


# =====================================================================================
# Closed-form Euler right-hand side
# =====================================================================================


def _euler_rhs(savings, experience, params, draws, weights, child_period):
    """Discounted expected marginal value of the (certain) death child.

    Death is consume-all, so the child consumes its whole beginning-of-period wealth and
    its marginal value is ``bequest'(wealth, experience)``. Experience is unchanged into
    the child (identity transition); the child's *own* age (``child_period``, not the
    proxy's last-period identity) sets its wage -- the quantity that makes the redirect
    load-bearing (see ``_budget``).

    """
    r = params["interest_rate"]
    wage = params["wage"] * (1 + params["wage_growth"] * child_period)
    marg_slope = 1 + params["bequest_exp_slope"] * experience
    rhs = 0.0
    for draw, weight in zip(draws, weights):
        income = np.exp(draw) * wage
        wealth_child = max((1 + r) * savings + income, params["consumption_floor"])
        rhs += weight * marg_slope / wealth_child
    return params["discount_factor"] * (1 + r) * rhs


def _assert_period0_euler_holds(model, solved):
    """Check every period-0 parent's Euler equation against the closed-form RHS.

    Shared by the oracle test and the regression test below, which monkeypatches the
    non-proxy child builder and expects this same check to fail.

    """
    quad_points, quad_weights = roots_sh_legendre(_MODEL_CONFIG["n_quad_points"])
    draws = (
        norm.ppf(quad_points) * _PARAMS["income_shock_std"]
        + _PARAMS["income_shock_mean"]
    )

    scs = model.model_structure["state_choice_space"]
    experience_grid = np.asarray(_MODEL_CONFIG["continuous_states"]["experience"])
    endog_grid = np.asarray(solved.endog_grid)
    policy = np.asarray(solved.policy)

    period_0_state_choices = np.where(scs[:, 0] == 0)[0]
    assert period_0_state_choices.size > 0

    checked = 0
    for state_choice_idx in period_0_state_choices:
        child_period = int(scs[state_choice_idx, 0]) + 1  # the death child's real age
        for experience_idx, experience in enumerate(experience_grid):
            for wealth_idx in range(endog_grid.shape[2]):
                resources = endog_grid[state_choice_idx, experience_idx, wealth_idx]
                consumption = policy[state_choice_idx, experience_idx, wealth_idx]
                if not np.isfinite(resources) or resources <= 0:
                    continue
                savings = resources - consumption
                lhs = 1 / consumption  # living parent's marginal utility
                rhs = _euler_rhs(
                    savings, experience, _PARAMS, draws, quad_weights, child_period
                )
                assert_allclose(lhs, rhs, atol=1e-6)
                checked += 1
    assert checked > 0


def test_death_proxy_satisfies_closed_form_euler():
    model, solved = _setup_and_solve()
    _assert_period0_euler_holds(model, solved)


def test_death_proxy_regression_detects_broken_redirect(monkeypatch):
    """The oracle above is only meaningful if it actually fails when the redirect is
    broken.

    Monkeypatch the non-proxy child builder to fall back to the proxy identity (the
    child the law of motion would see if the discrete-state decoupling in
    ``single_segment.py`` were undone) and check the same Euler equation now fails,
    since the death child's income would then be read at the proxy's (last) period
    instead of its own.

    """
    import dcegm.pre_processing.batches.single_segment as single_segment

    def _proxy_passthrough(
        state_space_dict,
        unique_child_states,
        state_row_for_state_choice,
        proxy_child_state_choices,
    ):
        return proxy_child_state_choices

    monkeypatch.setattr(
        single_segment, "build_no_proxy_child_state_choices", _proxy_passthrough
    )

    model, solved = _setup_and_solve()
    with np.testing.assert_raises(AssertionError):
        _assert_period0_euler_holds(model, solved)
