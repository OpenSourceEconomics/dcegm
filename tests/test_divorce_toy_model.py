"""Two-period (and n-period) dcegm divorce toy model: tests.

Model *mechanism* code lives in `tests/resources/divorce_model/` and takes
its numeric parametrization (`PARAMS`, grid sizes) as explicit arguments --
none of it is hardcoded there. This file owns the actual numbers:
  - `reference.py` -- independent hand-rolled backward-induction EGM solver
    (no dcegm, no upper envelope). Transition-based wealth rescaling: halve
    on divorce, double on marriage, unchanged otherwise; no multiplier in
    utility.
  - `dcegm_functions.py` -- the dcegm-facing model (utility, budget
    constraint, model setup). Per-period wealth/utility rescaling keyed off
    the *current* period's own partner_state only.

See the dcegm guide "Implementing a divorce/marriage transition without a
lagged partner state" (`docs/source/guides/`) for why these two
different-looking mechanisms are nonetheless the same underlying economics,
and for the general recipe.

Why `budget_constraint` (dcegm side) divides by the multiplier again
----------------------------------------------------------------------
dcegm's Euler-equation solver hardcodes the marginal return on savings as
`1 + params["interest_rate"]`
(`src/dcegm/egm/solve_euler_equation.py:166`,
`rhs_euler = marginal_utility_next * (1 + interest_rate) * discount_factor`).
It never differentiates the user-supplied `budget_constraint`, so any extra
multiplier applied to the incoming asset would be silently ignored in the
Euler equation's *price* of savings even though it's correctly reflected in
the wealth *level*. The fix: `budget_constraint` doubles the individual
asset if partnered, lets it earn interest, *and then divides the whole
expression by the same multiplier again*. For the asset term this is just
`mult * a * (1+r) / mult = a * (1+r)` -- the multiplier cancels exactly, so
`d(wealth)/d(a)` is `1+r` regardless of partner_state, matching what dcegm
assumes.
"""

import numpy as np
import pytest

from .resources.divorce_model import dcegm_functions as dm
from .resources.divorce_model import reference as ref

# ---------------------------------------------------------------------------
# Parametrization -- owned here, passed explicitly into every model/solver
# call. Nothing in tests/resources/divorce_model/ hardcodes any of this.
# ---------------------------------------------------------------------------

RHO = 0.8  # estimated mu_low = mu_high in est_params_alg1_sparse.pkl
DELTA = 0.5  # disutility of work
BETA = 0.96
R = 0.02
TASTE_SHOCK_SCALE = 0.2
Y_WORK = 30.0
Y_PARTNER = 20.0  # added whenever partnered *this* period, no lagged_choice
A_GRID_MAX = 300.0
A_GRID_POINTS = 200
PROB_MARRY = 0.3  # P(married next | single now)
PERSISTENCE_MARRIED = 0.8  # P(married next | married now)

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
    "y_partner": Y_PARTNER,
    "prob_marry": PROB_MARRY,
    "persistence_married": PERSISTENCE_MARRIED,
}

A_GRID = np.linspace(0.0, A_GRID_MAX, A_GRID_POINTS)
N_PERIODS = 4


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def dcegm_raw_arrays(model, solved, period, work0, partner_state_0):
    """Dcegm pads its stored arrays with extra points (e.g. a natural borrowing
    constraint point below the exogenous grid), so array *index* does not line up 1:1
    with the exogenous grid index -- clean and sort by the endogenous wealth *value*
    instead."""
    scs = model.model_structure["state_choice_space"]
    # columns: period, lagged_choice, partner_state, ..., choice (last)
    row = np.where(
        (scs[:, 0] == period)
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
# Two-period tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("choice", [0, 1])
@pytest.mark.parametrize("partner_state", [0, 1])
def test_dcegm_final_period_matches_analytic_formula(partner_state, choice):
    """Sanity check unaffected by the Euler-equation issue: the terminal
    period should just consume everything, value = felicity(scale*wealth,choice)."""
    model, solved = dm.build_and_solve(PARAMS, n_periods=2, a_grid=A_GRID)
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
    `a*(1+r)` regardless of partner_state.

    This is what makes dcegm's hardcoded `1+interest_rate` Euler-equation factor
    correct.

    """
    model, _ = dm.build_and_solve(PARAMS, n_periods=2, a_grid=A_GRID)
    compute_wealth = model.model_funcs["compute_assets_begin_of_period"]

    def wealth(asset, lagged_choice, partner_state):
        return compute_wealth(
            period=1,
            lagged_choice=lagged_choice,
            partner_state=partner_state,
            asset_end_of_previous_period=asset,
            income_shock_previous_period=0.0,
            params=PARAMS,
        )

    # The asset *return* (the slope of wealth in `a`) is exactly 1+r for
    # both partner states -- income (own and/or partner) is a constant
    # w.r.t. `a`, so it drops out of the slope regardless of its value.
    for partner_state in (0, 1):
        for lagged_choice in (0, 1):
            slope = (
                wealth(200.0, lagged_choice, partner_state)
                - wealth(100.0, lagged_choice, partner_state)
            ) / 100.0
            np.testing.assert_allclose(slope, 1 + R)

    # Own income (lagged_choice == 0) and partner income (partner_state == 1)
    # both still get divided by the multiplier when partnered.
    wealth_single_no_income = wealth(100.0, 1, 0)
    wealth_married_no_income = wealth(100.0, 1, 1)
    np.testing.assert_allclose(
        wealth_married_no_income - 100.0 * (1 + R), Y_PARTNER / 2
    )
    np.testing.assert_allclose(wealth_single_no_income, 100.0 * (1 + R))

    wealth_single_income = wealth(100.0, 0, 0)
    wealth_married_income = wealth(100.0, 0, 1)
    np.testing.assert_allclose(wealth_single_income - wealth_single_no_income, Y_WORK)
    np.testing.assert_allclose(
        wealth_married_income - wealth_married_no_income, Y_WORK / 2
    )


@pytest.mark.parametrize("partner_state", [0, 1])
@pytest.mark.parametrize("work0", [0, 1])
def test_dcegm_policy_matches_hand_solved_reference(partner_state, work0):
    """Cross-check dcegm's solved period-0 policy/value against the manual
    backward-induction reference, evaluated at dcegm's own native points --
    no interpolation on either side.

    dcegm's EGM identity `endog_grid = policy + a0_end` always holds for its
    own stored arrays, so `a0_end = endog_dcegm - policy_dcegm` recovers
    exactly which exogenous asset point produced each entry.

    dcegm's own state/policy live in *individual* terms; `utility_func`
    internally evaluates felicity at `scale * consumption`, i.e. at the
    *joint* quantity. So the reference (which has no such scale) has to be
    queried at `scale * a0_end` to land on the same joint quantity dcegm's
    utility implicitly uses. The resulting reference `policy` is then
    joint-scale and needs dividing by `scale` to compare against dcegm's
    individual-scale `policy`; `value` needs no such correction. Verified to
    match to floating-point precision (~1e-13) for all four
    (work0, partner_state) combinations.
    """
    model, solved = dm.build_and_solve(PARAMS, n_periods=2, a_grid=A_GRID)
    endog_dcegm, policy_dcegm, value_dcegm = dcegm_raw_arrays(
        model, solved, period=0, work0=work0, partner_state_0=partner_state
    )
    scale = 2.0 if partner_state == 1 else 1.0

    a0_end_dcegm = endog_dcegm - policy_dcegm
    # Skip near-zero a0_end: a degenerate corner (consumption_floor binds)
    # both sides handle slightly differently.
    keep = a0_end_dcegm > 1.0

    ref_policy = np.empty(keep.sum())
    ref_value = np.empty(keep.sum())
    for j, a0_end in enumerate(a0_end_dcegm[keep]):
        point = ref.solve_reference_at_point(
            scale * a0_end, partner_state, work0, PARAMS
        )
        ref_policy[j] = point["policy"] / scale
        ref_value[j] = point["value"]

    np.testing.assert_allclose(policy_dcegm[keep], ref_policy, rtol=1e-6)
    np.testing.assert_allclose(value_dcegm[keep], ref_value, rtol=1e-6)


# ---------------------------------------------------------------------------
# Four-period tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("period", list(range(N_PERIODS)))
@pytest.mark.parametrize("choice", [0, 1])
@pytest.mark.parametrize("partner_state", [0, 1])
def test_dcegm_policy_is_finite_and_monotone_for_n_periods(
    partner_state, choice, period
):
    """Basic sanity check that the n-period model solves cleanly at every
    period: policy should be finite, non-negative, and (weakly) increasing
    in wealth (more resources -> (weakly) more consumption)."""
    model, solved = dm.build_and_solve(PARAMS, n_periods=N_PERIODS, a_grid=A_GRID)
    endog, policy, value = dcegm_raw_arrays(
        model, solved, period=period, work0=choice, partner_state_0=partner_state
    )
    assert np.all(np.isfinite(policy))
    assert np.all(policy >= 0)
    assert np.all(np.diff(policy) >= -1e-8)  # allow tiny numerical noise
    assert np.all(np.isfinite(value[endog > 1.0]))


@pytest.mark.parametrize("partner_state", [0, 1])
@pytest.mark.parametrize("work0", [0, 1])
def test_dcegm_policy_matches_hand_solved_reference_n_periods(partner_state, work0):
    """Same cross-check as the two-period test, but for the full N_PERIODS-period model,
    comparing period 0's policy/value.

    Unlike the two-period case, periods before the terminal one are no
    longer analytic on the reference side either -- `solve_reference`
    interpolates its own stored grids for the continuation value, exactly
    as dcegm does internally, including two edge cases that must be handled
    explicitly rather than left to plain `np.interp` (which just clamps to
    a constant outside its domain, silently wrong in both directions):
    querying below the grid's own minimum (the borrowing constraint binds,
    consume everything) and above its maximum (linear extrapolation, since
    a marriage transition can double continuation wealth past what the
    grid was built to cover). See `solve_reference`'s
    `continuation_value_and_marg_util` for both. With those handled, this
    is no longer an exact floating-point match (periods before the terminal
    one are genuinely interpolated, on both sides, same as dcegm), but it's
    close: max relative error ~4e-4. See
    `test_dcegm_policy_matches_hand_solved_reference` above for the
    interpolation-free two-period version.

    """
    model, solved = dm.build_and_solve(PARAMS, n_periods=N_PERIODS, a_grid=A_GRID)
    endog_dcegm, policy_dcegm, value_dcegm = dcegm_raw_arrays(
        model, solved, period=0, work0=work0, partner_state_0=partner_state
    )
    scale = 2.0 if partner_state == 1 else 1.0

    a0_end_dcegm = endog_dcegm - policy_dcegm
    keep = a0_end_dcegm > 1.0

    # A married agent is queried at 2*a0_end (see the docstring above), which
    # can reach 2*A_GRID_MAX -- give the reference solve a wider grid than
    # dcegm's own so that the outer lookup below doesn't clamp at its top edge.
    wide_a_grid = np.linspace(0.0, 2 * A_GRID_MAX, 2 * A_GRID_POINTS)
    ref_solved = ref.solve_reference(N_PERIODS, PARAMS, wide_a_grid)
    ref_period0 = ref_solved[0][(partner_state, work0)]

    # ref_period0["policy"/"value"][i] is indexed by wide_a_grid[i] -- the
    # *exogenous asset* grid, not the endogenous wealth grid -- since that's
    # what solve_reference iterates over. Interpolate on that axis, same as
    # solve_reference_at_point conceptually does exactly (just off a
    # precomputed grid here instead of a fresh point solve).
    ref_policy = (
        np.interp(scale * a0_end_dcegm[keep], wide_a_grid, ref_period0["policy"])
        / scale
    )
    ref_value = np.interp(scale * a0_end_dcegm[keep], wide_a_grid, ref_period0["value"])

    # Actual max relative error is ~4e-4 (median ~3e-6) with the borrowing
    # constraint and top-of-grid extrapolation both handled explicitly in
    # `solve_reference` -- this tolerance has headroom, not a fudge factor.
    np.testing.assert_allclose(policy_dcegm[keep], ref_policy, rtol=1e-3)
    np.testing.assert_allclose(value_dcegm[keep], ref_value, rtol=1e-3, atol=1e-3)
