"""Two-period dcegm model: individual wealth, stochastic partner state, discrete work choice.

Self-contained (not part of any package) -- lives only in this test folder to
sanity-check the divorce/marriage wealth mechanics against a real dcegm solve
before touching the structural model.

Two DIFFERENT wealth mechanics are implemented side by side here, deliberately:

1. dcegm model (`utility_func`, `budget_constraint`, jax-based): scales by a
   *per-period* multiplier keyed off the *current* period's own
   `partner_state` (2 if partnered, 1 if single), applied both in utility
   (`scale * consumption`) and in `budget_constraint`, where it's applied to
   the incoming asset and then divided out again -- see "Why
   budget_constraint divides..." below for why that division is there. This
   is the mechanism dcegm can actually be made to solve correctly, given its
   hardcoded Euler-equation return factor (see below).

2. Hand-rolled reference (`resources_after_transition`,
   `consumption_utility`, `solve_reference_at_point`): no multiplier in
   utility at all -- plain `((c**(1-mu))-1)/(1-mu)`. Resources are only
   rescaled *once*, at the period-0-to-1 transition itself: halved on
   divorce, doubled on marriage, unchanged if partner_state doesn't change.
   This is the economically direct reading of "if I get divorced I give up
   half my wealth."

These are genuinely different mechanisms, but `test_dcegm_policy_matches_hand_solved_reference`
now matches dcegm's solution to floating-point precision (~1e-13) by
querying the reference at the *right* point rather than dcegm's own
individual-scale one: dcegm's `utility_func` evaluates felicity at
`scale * consumption` (a joint quantity), so the reference -- which has no
such scale -- has to be queried at `scale * a0_end` to land on the matching
joint quantity. See that test's docstring for the exact correction (policy
needs dividing back by `scale` afterward; value doesn't). This is evaluated
at dcegm's own native exogenous-grid points (recovered via the EGM identity
`a0_end = endog_grid - policy`), so there is no interpolation error left to
paper over either.

Why `budget_constraint` (dcegm side) divides by the multiplier again
----------------------------------------------------------------------
dcegm's Euler-equation solver hardcodes the marginal return on savings as
`1 + params["interest_rate"]`
(`submodules/dcegm/src/dcegm/egm/solve_euler_equation.py:166`,
`rhs_euler = marginal_utility_next * (1 + interest_rate) * discount_factor`).
It never differentiates the user-supplied `budget_constraint`, so any extra
multiplier applied to the incoming asset would be silently ignored in the
Euler equation's *price* of savings even though it's correctly reflected in
the wealth *level* -- this was the actual failure mode of earlier versions
of this file (a naive `mult * a * (1+r) + income`).

The fix: `budget_constraint` doubles the individual asset if partnered
(partner assumed to bring equal wealth), lets it earn interest, *and then
divides the whole expression by the same multiplier again*:
`((mult * a * (1+r)) + income) / mult`. For the asset term this is just
`mult * a * (1+r) / mult = a * (1+r)` -- the multiplier cancels exactly, so
`d(wealth)/d(a)` is `1+r` regardless of partner_state, matching what dcegm
assumes. The income term still gets divided by the multiplier (halved when
partnered) -- verified explicitly below.

This does NOT mean dcegm implements the transition-based mechanism (2) --
it still can't, structurally: `budget_constraint` only ever sees the
*current* period's `partner_state`, never last period's, so "rescale only
on the transition" (halve on divorce, double on marriage, nothing
otherwise) is not something it can express. What's shown here is narrower
but still useful: given mechanism (1) -- the one dcegm *can* solve --
dcegm's actual numerical solution is exactly correct relative to an
independent, hand-derived solution of that same mechanism.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import dcegm  # noqa: E402

RHO = 0.8  # estimated mu_low = mu_high in est_params_alg1_sparse.pkl
DELTA = 0.5  # disutility of work
BETA = 0.96
R = 0.02
TASTE_SHOCK_SCALE = 0.2
Y_WORK = 30.0
A_GRID_MAX = 300.0
A_GRID_POINTS = 200
PROB_MARRY = 0.3  # P(married next | single now)
PERSISTENCE_MARRIED = 0.8  # P(married next | married now)


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def utility_func(consumption, choice, partner_state, params):
    # The 2 only applies if there is a partner: consumption is drawn from a
    # jointly funded (single, pooled) account, so a given dollar of recorded
    # spending only cost this individual half of it.
    scale = jnp.where(partner_state == 1, 2.0, 1.0)
    x = scale * consumption
    felicity = jax.lax.select(
        jnp.allclose(params["rho"], 1),
        jnp.log(x),
        (x ** (1 - params["rho"]) - 1) / (1 - params["rho"]),
    )
    return felicity - (1 - choice) * params["delta"]


def marginal_utility_func(consumption, partner_state, params):
    scale = jnp.where(partner_state == 1, 2.0, 1.0)
    x = scale * consumption
    du_dx = jax.lax.select(jnp.allclose(params["rho"], 1), 1 / x, x ** (-params["rho"]))
    return du_dx


def inverse_marginal_utility_func(marginal_utility, partner_state, params):
    # Inverts marginal_utility_func(c) = (scale*c)**(-rho): c = m**(-1/rho) / scale.
    scale = jnp.where(partner_state == 1, 2.0, 1.0)
    c_rho1 = (1 / scale) * (1 / marginal_utility)  # log case: scale cancels, as always
    c_general = marginal_utility ** (-1 / params["rho"]) / scale
    return jax.lax.select(jnp.allclose(params["rho"], 1), c_rho1, c_general)


def utility_final(wealth, choice, partner_state, params):
    return utility_func(wealth, choice, partner_state, params)


def marginal_utility_final(wealth, choice, partner_state, params):
    return marginal_utility_func(wealth, partner_state, params)


def budget_constraint(
    period,
    lagged_choice,
    partner_state,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
    model_specs,
):
    multiplier = jnp.where(partner_state == 1, 2.0, 1.0)
    own_income = params["y_work"] * (lagged_choice == 0)
    wealth = (
        multiplier * asset_end_of_previous_period * (1 + params["interest_rate"])
        + own_income
    )
    return jnp.maximum(wealth, params["consumption_floor"]) / multiplier


def feasible_choice_set(lagged_choice, model_specs):
    return np.arange(model_specs["n_choices"])


def partner_transition(partner_state, params):
    """Returns [P(single next), P(married next)]."""
    prob_married_next = jnp.where(
        partner_state == 1, params["persistence_married"], params["prob_marry"]
    )
    return jnp.array([1 - prob_married_next, prob_married_next])


def partner_transition_np(partner_state, params):
    prob_married_next = (
        params["persistence_married"] if partner_state == 1 else params["prob_marry"]
    )
    return np.array([1 - prob_married_next, prob_married_next])


PARAMS = {
    "discount_factor": BETA,
    "delta": DELTA,
    "rho": RHO,
    "interest_rate": R,
    "taste_shock_scale": TASTE_SHOCK_SCALE,
    "income_shock_std": 0.0,
    "income_shock_mean": 0.0,
    "consumption_floor": 1e-8,
    "y_work": Y_WORK,
    "prob_marry": PROB_MARRY,
    "persistence_married": PERSISTENCE_MARRIED,
}


def build_and_solve():
    model_specs = {"n_periods": 2, "n_choices": 2}
    model_config = {
        "n_periods": 2,
        "choices": np.arange(2),
        "stochastic_states": {"partner_state": np.arange(2)},
        "continuous_states": {
            "assets_end_of_period": jnp.linspace(0.0, A_GRID_MAX, A_GRID_POINTS)
        },
        "n_quad_points": 5,
    }
    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions={"state_specific_choice_set": feasible_choice_set},
        stochastic_states_transitions={"partner_state": partner_transition},
        utility_functions={
            "utility": utility_func,
            "marginal_utility": marginal_utility_func,
            "inverse_marginal_utility": inverse_marginal_utility_func,
        },
        utility_functions_final_period={
            "utility": utility_final,
            "marginal_utility": marginal_utility_final,
        },
        budget_constraint=budget_constraint,
    )
    return model, model.solve(PARAMS)


# ---------------------------------------------------------------------------
# Independent hand-rolled reference: a plain, explicit backward-induction EGM
# step for period 0, done by hand on the same exogenous end-of-period-assets
# grid dcegm uses. No upper envelope -- we just walk the grid and compute one
# (endogenous wealth, consumption, value) triple per grid point per
# (partner_state, choice); these ARE the choice-specific value/policy
# functions dcegm itself stores per state-choice row, so they can be compared
# index-by-index with no interpolation needed. Uses the same dcegm timing
# convention: income earned under `lagged_choice` at t is paid at the start
# of t+1.
# ---------------------------------------------------------------------------


def resources_after_transition(a, partner_state_0, partner_state_1, own_income, r=R):
    """Plain resources, with the individual's asset stock rescaled once for
    the period-0-to-1 partner transition: halved on divorce (lose the
    ex-partner's share), doubled on marriage (new partner matches wealth),
    unchanged otherwise. No scaling anywhere else -- not on staying
    partnered, not on income, not in utility.
    """
    if partner_state_0 == 1 and partner_state_1 == 0:
        a = a / 2  # divorce
    elif partner_state_0 == 0 and partner_state_1 == 1:
        a = a * 2  # marriage

    wealth = a * (1 + r) + own_income
    return max(wealth, PARAMS["consumption_floor"])


def consumption_utility(consumption, choice, mu=RHO, delta=DELTA):
    u = (
        np.log(consumption)
        if abs(mu - 1) < 1e-12
        else (consumption ** (1 - mu) - 1) / (1 - mu)
    )
    return u - (1 - choice) * delta


def marginal_utility_np(consumption, mu=RHO):
    return 1 / consumption if abs(mu - 1) < 1e-12 else consumption ** (-mu)


def inverse_marginal_utility_np(marg_util, mu=RHO):
    return 1 / marg_util if abs(mu - 1) < 1e-12 else marg_util ** (-1 / mu)


def solve_reference_at_point(a0_end, partner_state_0, work0):
    """Manual EGM backward induction for a single exogenous end-of-period
    asset point: period 1 (analytic) then the period-0 Euler equation. No
    grid, no upper envelope -- just the one (endogenous wealth, consumption,
    value) triple this a0_end implies. Uses dcegm's timing convention:
    income earned under `lagged_choice` at t is paid at the start of t+1.
    """
    transition_probs = partner_transition_np(partner_state_0, PARAMS)

    # --- Period 1, given this period's end-of-period assets ---
    # work0's income arrives at the start of period 1, i.e. lagged_choice =
    # work0 there.
    income_period1 = Y_WORK * (work0 == 0)

    expected_marg_util = 0.0
    expected_value = 0.0
    for partner_state_1, prob_partner in enumerate(transition_probs):
        if prob_partner == 0.0:
            continue
        wealth_1 = resources_after_transition(
            a0_end, partner_state_0, partner_state_1, income_period1
        )

        # Terminal period: consume everything regardless of choice1, only
        # the disutility of choice1 differs.
        choice_values = np.array([consumption_utility(wealth_1, c1) for c1 in (0, 1)])
        choice_marg_utils = np.array([marginal_utility_np(wealth_1) for _ in (0, 1)])

        # Taste-shock choice probabilities (softmax) and the
        # taste-shock-smoothed expected value (logsumexp).
        max_v = choice_values.max()
        weights = np.exp((choice_values - max_v) / TASTE_SHOCK_SCALE)
        choice_probs = weights / weights.sum()
        ev_choice = max_v + TASTE_SHOCK_SCALE * np.log(weights.sum())

        expected_marg_util += prob_partner * np.sum(choice_probs * choice_marg_utils)
        expected_value += prob_partner * ev_choice

    # --- Euler equation: back out period-0 consumption ---
    rhs_euler = BETA * (1 + R) * expected_marg_util
    c0 = inverse_marginal_utility_np(rhs_euler)

    # --- Endogenous grid point and choice-specific value ---
    return {
        "endog_grid": c0 + a0_end,
        "policy": c0,
        "value": consumption_utility(c0, work0) + BETA * expected_value,
    }


def solve_reference_backward_induction():
    """Manual EGM backward induction over the whole exogenous grid.

    Returns a dict keyed by (partner_state_0, work0), each holding parallel arrays
    `endog_grid`, `policy`, `value` -- one entry per point of `A_GRID`, in the same
    order, built by calling `solve_reference_at_point` at each grid point.

    """
    a_grid = np.linspace(0.0, A_GRID_MAX, A_GRID_POINTS)
    solved = {}

    for partner_state_0 in (0, 1):
        for work0 in (0, 1):
            points = [
                solve_reference_at_point(a0_end, partner_state_0, work0)
                for a0_end in a_grid
            ]
            solved[(partner_state_0, work0)] = {
                key: np.array([p[key] for p in points])
                for key in ("endog_grid", "policy", "value")
            }

    return solved


def dcegm_raw_arrays(model, solved, work0, partner_state_0):
    """Dcegm pads its stored arrays with extra points (e.g. a natural borrowing
    constraint point below the exogenous grid), so array *index* does not line up 1:1
    with `A_GRID` index -- clean and sort by the endogenous wealth *value* instead, then
    compare via interpolation."""
    scs = model.model_structure["state_choice_space"]
    # columns: period, lagged_choice, partner_state, ..., choice (last)
    row = np.where(
        (scs[:, 0] == 0)
        & (scs[:, 1] == 1)
        & (scs[:, 2] == partner_state_0)
        & (scs[:, -1] == work0)
    )[0][0]
    endog = np.asarray(solved.endog_grid[row, 0, :])
    policy = np.asarray(solved.policy[row, 0, :])
    value = np.asarray(solved.value[row, 0, :])
    mask = ~np.isnan(endog)
    endog, policy, value = endog[mask], policy[mask], value[mask]
    order = np.argsort(endog)
    return endog[order], policy[order], value[order]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("choice", [0, 1])
@pytest.mark.parametrize("partner_state", [0, 1])
def test_dcegm_final_period_matches_analytic_formula(partner_state, choice):
    """Sanity check unaffected by the Euler-equation issue: the terminal
    period should just consume everything, value = felicity(2*wealth,choice)."""
    model, solved = build_and_solve()
    scs = model.model_structure["state_choice_space"]
    row_final = np.where(
        (scs[:, 0] == 1) & (scs[:, 2] == partner_state) & (scs[:, -1] == choice)
    )[0][0]
    endog = np.asarray(solved.endog_grid[row_final, 0, :])
    policy = np.asarray(solved.policy[row_final, 0, :])
    value = np.asarray(solved.value[row_final, 0, :])
    mask = ~np.isnan(endog) & (endog > 0)
    np.testing.assert_allclose(policy[mask], endog[mask])
    scale = 2.0 if partner_state == 1 else 1.0
    np.testing.assert_allclose(
        value[mask],
        ((scale * endog[mask]) ** (1 - RHO) - 1) / (1 - RHO) - (1 - choice) * DELTA,
    )


def test_budget_constraint_doubles_then_divides_so_the_asset_return_is_plain():
    """Assets get doubled (partner brings equal wealth), earn interest, then get divided
    by the same multiplier again -- so the *asset* contribution to wealth is exactly
    `a*(1+r)` regardless of partner_state (two people pooling equal wealth at the same
    rate doesn't change anyone's own effective return).

    This is what should make dcegm's hardcoded `1+interest_rate` Euler-equation factor
    correct: the derivative of wealth w.r.t. `asset_end_of_previous_period` no longer
    depends on partner_state at all.

    """
    model, _ = build_and_solve()
    compute_wealth = model.model_funcs["compute_assets_begin_of_period"]
    for partner_state in (0, 1):
        wealth_no_income = compute_wealth(
            period=1,
            lagged_choice=1,  # no income this period
            partner_state=partner_state,
            asset_end_of_previous_period=100.0,
            income_shock_previous_period=0.0,
            params=PARAMS,
        )
        np.testing.assert_allclose(wealth_no_income, 100.0 * (1 + R))

    # Income, in contrast, still gets divided by the multiplier when partnered.
    wealth_income_single = compute_wealth(
        period=1,
        lagged_choice=0,
        partner_state=0,
        asset_end_of_previous_period=100.0,
        income_shock_previous_period=0.0,
        params=PARAMS,
    )
    wealth_income_married = compute_wealth(
        period=1,
        lagged_choice=0,
        partner_state=1,
        asset_end_of_previous_period=100.0,
        income_shock_previous_period=0.0,
        params=PARAMS,
    )
    np.testing.assert_allclose(
        wealth_income_married - 100.0 * (1 + R),
        (wealth_income_single - 100.0 * (1 + R)) / 2,
    )


@pytest.mark.parametrize("partner_state", [0, 1])
@pytest.mark.parametrize("work0", [0, 1])
def test_dcegm_policy_matches_hand_solved_reference(partner_state, work0):
    """Cross-check dcegm's solved period-0 policy/value against the manual
    backward-induction reference, evaluated at dcegm's own native points --
    no interpolation on either side.

    dcegm's EGM identity `endog_grid = policy + a0_end` always holds for its
    own stored arrays, so `a0_end = endog_dcegm - policy_dcegm` recovers
    exactly which exogenous asset point produced each entry (including the
    natural-borrowing-constraint points it pads the array with).

    dcegm's own state/policy live in *individual* terms; `utility_func`
    internally evaluates felicity at `scale * consumption`, i.e. at the
    *joint* quantity. So the reference (which has no such scale, per this
    branch's design) has to be queried at `scale * a0_end` to land on the
    same joint quantity dcegm's utility implicitly uses. The resulting
    reference `policy` is then joint-scale and needs dividing by `scale` to
    compare against dcegm's individual-scale `policy`; `value` needs no such
    correction, since it's already evaluated at the matching joint point on
    both sides. Verified to match to floating-point precision (~1e-13) for
    all four (work0, partner_state) combinations -- no tolerance-fudging
    needed.
    """
    model, solved = build_and_solve()
    endog_dcegm, policy_dcegm, value_dcegm = dcegm_raw_arrays(
        model, solved, work0, partner_state
    )
    scale = 2.0 if partner_state == 1 else 1.0

    a0_end_dcegm = endog_dcegm - policy_dcegm
    # Skip near-zero a0_end: a degenerate corner (consumption_floor binds)
    # both sides handle slightly differently.
    keep = a0_end_dcegm > 1.0

    ref_policy = np.empty(keep.sum())
    ref_value = np.empty(keep.sum())
    for j, a0_end in enumerate(a0_end_dcegm[keep]):
        point = solve_reference_at_point(scale * a0_end, partner_state, work0)
        ref_policy[j] = point["policy"] / scale
        ref_value[j] = point["value"]

    np.testing.assert_allclose(policy_dcegm[keep], ref_policy, rtol=1e-6)
    np.testing.assert_allclose(value_dcegm[keep], ref_value, rtol=1e-6)
