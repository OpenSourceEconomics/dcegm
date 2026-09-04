"""Absolute checks: dcegm against independently implemented solutions.

Every other test for the features on this branch is *relative* -- one dcegm solve
compared against another dcegm solve (two configurations that must agree, or two
evaluation granularities that must agree). Those catch regressions, but share a
blind spot: an error made identically on both sides is invisible to them.

The tests here pin dcegm against solutions computed outside it, in two flavours:

**Closed form** (first half). A two-period model where the Euler equation is exact:
period 1 is terminal, so consumption equals wealth and the marginal utility of
wealth is ``wealth ** -rho`` regardless of the period-1 choice. Period 0 then
satisfies, for each savings point ``a`` on that state-choice's *own* grid::

    RHS      = discount_factor * (1 + r) * E_shock[ sum_k P(k) * wealth_1(k) ** -rho ]
    c_0      = RHS ** (-1 / rho)
    wealth_0 = a + c_0

with ``P(k)`` the logit choice probabilities over period-1 choices (see
``aggregate_marg_utils_and_exp_values``).

**Hand-solved reference** (second half). The divorce model's
``tests/resources/divorce_model/reference.py`` -- a plain-NumPy backward-induction
EGM sharing no code with the solver -- over four periods with stochastic partner
transitions. Its ``grid_for`` mirrors dcegm's ``continuous_grid_functions``.

Two independent properties get checked throughout, and neither substitutes for the
other:

1. **Which grid was used** -- ``endog_grid - policy`` recovers the end-of-period
   assets grid a state-choice actually solved on. This is the load-bearing check
   for grid threading; the value comparisons below provably cannot make it.
2. **Whether the economics are right** -- the Euler residual, or the policy/value
   comparison against the reference.

Not covered here: the Druedahl-Jorgensen ``assets_begin_of_period`` grid. There the
stored policy is *interpolated* onto the common wealth grid rather than being a raw
EGM solution, so it does not satisfy the Euler equation pointwise and has no closed
form of this kind. It is covered by the equivalence tests in
``test_state_specific_grids_end_to_end.py``.

"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm

import dcegm
import tests.resources.divorce_model.dcegm_functions as dm
import tests.resources.divorce_model.reference as ref
from tests.test_divorce_toy_model import (
    A_GRID_MAX,
    A_GRID_POINTS,
    N_PERIODS,
)
from tests.test_divorce_toy_model import PARAMS as DIVORCE_PARAMS
from tests.test_divorce_toy_model import dcegm_raw_arrays

# =====================================================================================
# Part 1: closed-form Euler equation, two-period model
# =====================================================================================


RHO = 1.5
N_ASSET_POINTS = 12
N_QUAD = 5

CLOSED_FORM_PARAMS = {
    "discount_factor": 0.95,
    "interest_rate": 0.04,
    "rho": RHO,
    "delta": 0.3,
    "taste_shock_scale": 0.2,
    "income_shock_std": 0.2,
    "income_shock_mean": 0.0,
}

MODEL_SPECS = {"n_choices": 2, "wage": 5.0, "pension": 2.0, "work_cost": 0.4}

# Declared default grid. Its *values* are unused once a continuous_grid_functions
# entry is given, but assets_end_of_period must still declare a real array (it is
# the one name exempt from the None convention, see process_continuous_grid_functions)
# and that array fixes the expected length every state-choice's own grid is
# validated against.
DEFAULT_ASSET_GRID = np.linspace(0.0, 20.0, N_ASSET_POINTS)


# =====================================================================================
# Model functions
# =====================================================================================


def _utility(consumption, choice, params):
    return consumption ** (1 - params["rho"]) / (1 - params["rho"]) - params[
        "delta"
    ] * (choice == 0)


def _marginal_utility(consumption, params):
    return consumption ** (-params["rho"])


def _inverse_marginal_utility(marginal_utility, params):
    return marginal_utility ** (-1 / params["rho"])


def _utility_final(wealth, choice, params):
    return wealth ** (1 - params["rho"]) / (1 - params["rho"]) - params["delta"] * (
        choice == 0
    )


def _marginal_utility_final(wealth, params):
    return wealth ** (-params["rho"])


UTILITY_FUNCTIONS = {
    "utility": _utility,
    "marginal_utility": _marginal_utility,
    "inverse_marginal_utility": _inverse_marginal_utility,
}
UTILITY_FUNCTIONS_FINAL = {
    "utility": _utility_final,
    "marginal_utility": _marginal_utility_final,
}


def _budget(
    lagged_choice,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
    model_specs,
):
    """Beginning-of-period wealth.

    Does not depend on the current choice.

    """
    income = jnp.where(lagged_choice == 0, model_specs["wage"], model_specs["pension"])
    return (
        (1 + params["interest_rate"]) * asset_end_of_previous_period
        + income
        + income_shock_previous_period
    )


def _budget_choice_dependent(
    lagged_choice,
    choice,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
    model_specs,
):
    """Same, plus a cost paid when this period's own choice is to work.

    Declaring ``choice`` is what routes the solve down the per-state-choice law of
    motion (see _transition_funcs_depend_on_choice), and the subtracted cost is what
    makes that routing observable in the solution.

    """
    base = _budget(
        lagged_choice=lagged_choice,
        asset_end_of_previous_period=asset_end_of_previous_period,
        income_shock_previous_period=income_shock_previous_period,
        params=params,
        model_specs=model_specs,
    )
    return base - model_specs["work_cost"] * (choice == 0)


def _assets_grid_by_group(group):
    """A genuinely different end-of-period assets grid per group."""
    return jnp.asarray(DEFAULT_ASSET_GRID) * (1.0 + 0.5 * group)


# =====================================================================================
# Closed form
# =====================================================================================


def _quadrature():
    quad_points, quad_weights = roots_sh_legendre(N_QUAD)
    quad_draws = (
        norm.ppf(quad_points) * CLOSED_FORM_PARAMS["income_shock_std"]
        + CLOSED_FORM_PARAMS["income_shock_mean"]
    )
    return np.asarray(quad_draws), np.asarray(quad_weights)


def _closed_form_consumption(savings, period_zero_choice, choices, budget_fn):
    """Period-0 consumption implied by the Euler equation, for one savings point.

    ``period_zero_choice`` becomes the period-1 state's ``lagged_choice``. The inner
    loop is over period-1 choices, which matter only when the budget itself depends on
    the current choice; otherwise every ``k`` gives the same wealth and the logit
    weights sum back to one.

    """
    quad_draws, quad_weights = _quadrature()

    wealth_next = np.empty((len(choices), len(quad_draws)))
    for k_idx, choice_next in enumerate(choices):
        wealth_next[k_idx] = np.asarray(
            budget_fn(
                lagged_choice=period_zero_choice,
                choice=choice_next,
                asset_end_of_previous_period=savings,
                income_shock_previous_period=quad_draws,
                params=CLOSED_FORM_PARAMS,
                model_specs=MODEL_SPECS,
            )
        )

    values_next = np.empty_like(wealth_next)
    for k_idx, choice_next in enumerate(choices):
        values_next[k_idx] = np.asarray(
            _utility_final(
                wealth=wealth_next[k_idx], choice=choice_next, params=CLOSED_FORM_PARAMS
            )
        )
    marg_util_next = wealth_next ** (-RHO)

    # Logit choice probabilities over period-1 choices, rescaled by the max the
    # same way calculate_choice_probs_and_unsqueezed_logsum does.
    scale = CLOSED_FORM_PARAMS["taste_shock_scale"]
    rescaled = np.exp((values_next - values_next.max(axis=0, keepdims=True)) / scale)
    choice_probs = rescaled / rescaled.sum(axis=0, keepdims=True)

    marg_util_aggregated = (choice_probs * marg_util_next).sum(axis=0)
    marg_util_integrated = marg_util_aggregated @ quad_weights

    rhs = (
        marg_util_integrated
        * (1 + CLOSED_FORM_PARAMS["interest_rate"])
        * CLOSED_FORM_PARAMS["discount_factor"]
    )
    return rhs ** (-1 / RHO)


def _budget_ignoring_choice(**kwargs):
    kwargs.pop("choice", None)
    return _budget(**kwargs)


# =====================================================================================
# Model setup
# =====================================================================================


def _solve(choices, budget_fn, continuous_grid_functions=None):
    model_config = {
        "n_periods": 2,
        "choices": choices,
        "deterministic_states": {"group": [0, 1]},
        "continuous_states": {"assets_end_of_period": DEFAULT_ASSET_GRID},
        "n_quad_points": N_QUAD,
    }
    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=MODEL_SPECS,
        utility_functions=UTILITY_FUNCTIONS,
        utility_functions_final_period=UTILITY_FUNCTIONS_FINAL,
        budget_constraint=budget_fn,
        continuous_grid_functions=continuous_grid_functions,
    )
    return model, model.solve(CLOSED_FORM_PARAMS)


def _period_zero_state_choices(model):
    state_choice_space = model.model_structure["state_choice_space"]
    names = model.model_structure["discrete_states_names"] + ["choice"]
    idxs = np.where(state_choice_space[:, 0] == 0)[0]
    rows = [
        {name: int(state_choice_space[i, col]) for col, name in enumerate(names)}
        for i in idxs
    ]
    return idxs, rows


# =====================================================================================
# Tests
# =====================================================================================


def test_state_specific_assets_grid_matches_closed_form():
    """The headline case: a per-group end-of-period assets grid.

    Checks both that each state-choice solved on *its own* grid, and that the resulting
    consumption satisfies the closed-form Euler equation on that grid.

    """
    choices = [0, 1]
    model, solved = _solve(
        choices=choices,
        budget_fn=_budget,
        continuous_grid_functions={"assets_end_of_period": _assets_grid_by_group},
    )
    endog_grid = np.asarray(solved.endog_grid)
    policy = np.asarray(solved.policy)

    idxs, rows = _period_zero_state_choices(model)
    n_checked = 0

    for state_choice_idx, row in zip(idxs, rows):
        own_grid = np.asarray(_assets_grid_by_group(group=row["group"]))

        # Column 0 is the zero-wealth point prepended by the solver; the raw EGM
        # points follow.
        grid_row = endog_grid[state_choice_idx, 0, 1 : len(own_grid) + 1]
        policy_row = policy[state_choice_idx, 0, 1 : len(own_grid) + 1]

        # 1) endog_grid = savings + consumption, so this recovers the savings grid
        #    the state-choice actually solved on. A wrong grid shows up here even
        #    though the Euler residual below would not notice.
        implied_savings = grid_row - policy_row
        assert_allclose(implied_savings, own_grid, atol=1e-6)

        # 2) The economics, against the independently computed Euler equation.
        for point, (savings, consumption) in enumerate(zip(own_grid, policy_row)):
            if savings <= 1e-10:
                continue  # zero-savings point: corner, not an interior Euler solution
            expected = _closed_form_consumption(
                savings=savings,
                period_zero_choice=row["choice"],
                choices=choices,
                budget_fn=_budget_ignoring_choice,
            )
            assert_allclose(
                consumption, expected, rtol=1e-6, err_msg=f"{row} @ {point}"
            )
            n_checked += 1

    assert n_checked > 0


def test_default_grid_matches_closed_form_without_grid_functions():
    """Sensitivity anchor: the same closed form must pin the *unmodified* solver.

    If this failed, a failure of the test above could not be attributed to the state-
    specific grid machinery -- it would just mean the closed form is wrong.

    """
    choices = [0, 1]
    model, solved = _solve(choices=choices, budget_fn=_budget)
    endog_grid = np.asarray(solved.endog_grid)
    policy = np.asarray(solved.policy)

    idxs, rows = _period_zero_state_choices(model)

    for state_choice_idx, row in zip(idxs, rows):
        grid_row = endog_grid[state_choice_idx, 0, 1 : N_ASSET_POINTS + 1]
        policy_row = policy[state_choice_idx, 0, 1 : N_ASSET_POINTS + 1]
        assert_allclose(grid_row - policy_row, DEFAULT_ASSET_GRID, atol=1e-6)

        for savings, consumption in zip(DEFAULT_ASSET_GRID, policy_row):
            if savings <= 1e-10:
                continue
            expected = _closed_form_consumption(
                savings=savings,
                period_zero_choice=row["choice"],
                choices=choices,
                budget_fn=_budget_ignoring_choice,
            )
            assert_allclose(consumption, expected, rtol=1e-6)


def test_choice_dependent_budget_matches_closed_form():
    """A budget equation that reads ``choice`` must be solved with it.

    The closed form here aggregates over period-1 choices with logit weights, because
    the budget makes next period's wealth differ by that choice. Solving while ignoring
    ``choice`` would produce a different consumption policy -- which is exactly what the
    companion test below confirms.

    """
    choices = [0, 1]
    model, solved = _solve(choices=choices, budget_fn=_budget_choice_dependent)
    assert model.model_funcs["transition_funcs_depend_on_choice"]

    endog_grid = np.asarray(solved.endog_grid)
    policy = np.asarray(solved.policy)
    idxs, rows = _period_zero_state_choices(model)

    n_checked = 0
    for state_choice_idx, row in zip(idxs, rows):
        grid_row = endog_grid[state_choice_idx, 0, 1 : N_ASSET_POINTS + 1]
        policy_row = policy[state_choice_idx, 0, 1 : N_ASSET_POINTS + 1]
        assert_allclose(grid_row - policy_row, DEFAULT_ASSET_GRID, atol=1e-6)

        for savings, consumption in zip(DEFAULT_ASSET_GRID, policy_row):
            if savings <= 1e-10:
                continue
            expected = _closed_form_consumption(
                savings=savings,
                period_zero_choice=row["choice"],
                choices=choices,
                budget_fn=_budget_choice_dependent,
            )
            assert_allclose(consumption, expected, rtol=1e-6)
            n_checked += 1

    assert n_checked > 0


def test_choice_dependent_closed_form_differs_from_choice_free_one():
    """Sensitivity anchor for the test above.

    Confirms the choice-dependent closed form is genuinely a different number, so that
    test would fail if the solver silently dropped ``choice`` from the budget equation.

    """
    savings = 5.0
    with_choice = _closed_form_consumption(
        savings=savings,
        period_zero_choice=0,
        choices=[0, 1],
        budget_fn=_budget_choice_dependent,
    )
    without_choice = _closed_form_consumption(
        savings=savings,
        period_zero_choice=0,
        choices=[0, 1],
        budget_fn=_budget_ignoring_choice,
    )
    assert not np.isclose(with_choice, without_choice)


def test_state_specific_grid_and_choice_dependent_budget_together():
    """Both new features at once, still against the closed form.

    This combination routes through the per-state-choice law of motion *and* the per-
    state-choice grid selection simultaneously, which no other test covers.

    """
    choices = [0, 1]
    model, solved = _solve(
        choices=choices,
        budget_fn=_budget_choice_dependent,
        continuous_grid_functions={"assets_end_of_period": _assets_grid_by_group},
    )
    endog_grid = np.asarray(solved.endog_grid)
    policy = np.asarray(solved.policy)
    idxs, rows = _period_zero_state_choices(model)

    n_checked = 0
    for state_choice_idx, row in zip(idxs, rows):
        own_grid = np.asarray(_assets_grid_by_group(group=row["group"]))
        grid_row = endog_grid[state_choice_idx, 0, 1 : len(own_grid) + 1]
        policy_row = policy[state_choice_idx, 0, 1 : len(own_grid) + 1]
        assert_allclose(grid_row - policy_row, own_grid, atol=1e-6)

        for savings, consumption in zip(own_grid, policy_row):
            if savings <= 1e-10:
                continue
            expected = _closed_form_consumption(
                savings=savings,
                period_zero_choice=row["choice"],
                choices=choices,
                budget_fn=_budget_choice_dependent,
            )
            assert_allclose(consumption, expected, rtol=1e-6)
            n_checked += 1

    assert n_checked > 0


# =====================================================================================
# Part 2: hand-solved n-period reference (divorce model)
# =====================================================================================


BASE_GRID = np.linspace(0.0, 2 * A_GRID_MAX, 2 * A_GRID_POINTS)


def _grid_scale(period, choice):
    """Scale factor making each (period, choice) own a genuinely different grid."""
    return 1.0 + 0.20 * period + 0.35 * choice


def assets_grid(period, choice):
    """The state-choice-specific grid, used by *both* sides.

    dcegm calls this with keyword arguments filtered to its signature; the reference
    calls it through ``ref.grid_for``, also by keyword. Sharing one definition removes
    any chance of the two sides silently drifting apart -- which would show up as a
    comparison failure that looks like a solver bug.

    ``period`` and ``choice`` are the only admissible dependencies here; see
    ``ref.grid_for`` for why ``partner_state`` and ``lagged_choice`` are not.

    """
    return BASE_GRID * _grid_scale(period, choice)


def _solve_both():
    model, solved = dm.build_and_solve(
        DIVORCE_PARAMS,
        n_periods=N_PERIODS,
        a_grid=BASE_GRID,
        continuous_grid_functions={"assets_end_of_period": assets_grid},
    )
    ref_solved = ref.solve_reference(N_PERIODS, DIVORCE_PARAMS, assets_grid)
    return model, solved, ref_solved


@pytest.mark.parametrize("work0", [0, 1])
@pytest.mark.parametrize("partner_state", [0, 1])
def test_state_specific_grid_matches_hand_solved_reference(partner_state, work0):
    """Dcegm with a per-(period, choice) assets grid must reproduce the reference.

    What this establishes: with state-specific grids active, dcegm still solves
    the *right model* -- checked against an implementation sharing no code with
    the solver, over 4 periods with stochastic partner transitions.

    What it deliberately does not establish: *which* grid was used. Measured
    directly (see the note below), a reference solved on the wrong -- shared --
    grid lands within 2.3e-4 to 4.5e-3 relative of dcegm, while the correct-grid
    reference lands within 1.2e-4 to 4.5e-4; the ranges overlap, so no tolerance
    separates them. That is inherent rather than a weakness of the setup: the
    policy is the same underlying function of wealth either way, and sampling it
    on a different grid then interpolating recovers nearly the same values.
    ``test_dcegm_solves_on_its_own_declared_grid`` below is what pins the grid.

    """
    model, solved, ref_solved = _solve_both()

    endog_dcegm, policy_dcegm, value_dcegm = dcegm_raw_arrays(
        model, solved, period=0, work0=work0, partner_state_0=partner_state
    )
    # The divorce model stores individual wealth; a married agent's reference
    # solve is in household units, hence the factor of two (same convention as
    # test_dcegm_policy_matches_hand_solved_reference_n_periods).
    scale = 2.0 if partner_state == 1 else 1.0

    # Restrict to end-of-period assets whose *continuation* lookup next period is
    # genuinely interpolated on both sides. A married agent is queried at
    # 2*a_end against period 1's own grid, which tops out at
    # BASE_GRID.max() * _grid_scale(1, choice); past that both dcegm and the
    # reference extrapolate, by different rules, and comparing there would be
    # measuring the extrapolation rules rather than the solve.
    interior_max = BASE_GRID.max() * _grid_scale(1, 0) / scale
    a0_end_dcegm = endog_dcegm - policy_dcegm
    keep = (a0_end_dcegm > 1.0) & (a0_end_dcegm < 0.8 * interior_max)
    assert keep.sum() > 50

    ref_period0 = ref_solved[0][(partner_state, work0)]
    ref_own_grid = ref.grid_for(assets_grid, 0, work0)

    ref_policy = (
        np.interp(scale * a0_end_dcegm[keep], ref_own_grid, ref_period0["policy"])
        / scale
    )
    ref_value = np.interp(
        scale * a0_end_dcegm[keep], ref_own_grid, ref_period0["value"]
    )

    np.testing.assert_allclose(policy_dcegm[keep], ref_policy, rtol=1e-3)
    np.testing.assert_allclose(value_dcegm[keep], ref_value, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("work0", [0, 1])
@pytest.mark.parametrize("partner_state", [0, 1])
def test_dcegm_solves_on_its_own_declared_grid(partner_state, work0):
    """The savings points dcegm actually used are that state-choice's own grid.

    ``endog_grid = savings + consumption``, so ``endog_grid - policy`` recovers
    the end-of-period assets grid the state-choice was solved on. This is the
    load-bearing test for grid threading, and unlike the policy comparison above
    it discriminates sharply: solving the same model *without* the grid functions
    leaves 385 of this state-choice's 400 own grid points unaccounted for.

    """
    model, solved, _ = _solve_both()
    endog_dcegm, policy_dcegm, _ = dcegm_raw_arrays(
        model, solved, period=0, work0=work0, partner_state_0=partner_state
    )
    own_grid = np.asarray(assets_grid(period=0, choice=work0))

    implied_savings = endog_dcegm - policy_dcegm
    # dcegm may prepend a natural-borrowing-constraint point below the exogenous
    # grid, so match the declared grid as a subset rather than elementwise.
    for a_end in own_grid:
        assert np.isclose(
            implied_savings, a_end, atol=1e-6
        ).any(), f"savings point {a_end} of this state-choice's own grid was not solved"


def test_grid_varying_only_by_period_and_choice_is_actually_state_specific():
    """Sensitivity anchor: confirm the grids really do differ across state-choices.

    Without this, the two tests above would still pass if ``assets_grid`` happened to
    collapse to one shared grid -- they would just be re-testing the default path.

    """
    grid_p0_work = np.asarray(assets_grid(period=0, choice=0))
    grid_p0_retire = np.asarray(assets_grid(period=0, choice=1))
    grid_p1_work = np.asarray(assets_grid(period=1, choice=0))

    # Varies with choice, and with period.
    assert not np.allclose(grid_p0_work, grid_p0_retire)
    assert not np.allclose(grid_p0_work, grid_p1_work)
    # At least one state-choice differs from the declared default, so the solve
    # cannot be silently falling back to it.
    assert not np.allclose(grid_p0_retire, BASE_GRID)


def test_grid_depending_on_stochastic_partner_state_is_rejected():
    """A grid varying with the stochastic partner_state must not be accepted.

    Both partner states are parents of the same children here, so they would disagree on
    the grid feeding a shared continuation value. dcegm has to reject this at build time
    rather than silently pick one -- documenting the boundary of what the reference
    above is allowed to mirror.

    """

    def bad_grid(period, partner_state):
        return BASE_GRID * (1.0 + 0.5 * partner_state)

    with pytest.raises(ValueError, match="different grids"):
        dm.build_and_solve(
            DIVORCE_PARAMS,
            n_periods=N_PERIODS,
            a_grid=BASE_GRID,
            continuous_grid_functions={"assets_end_of_period": bad_grid},
        )
