"""State-choice-specific assets grids checked against the hand-solved reference.

The other tests for this feature are *relative*: one dcegm solve against another
dcegm solve. They cannot catch an error made identically on both sides. This file
instead checks dcegm against ``tests/resources/divorce_model/reference.py`` -- an
independent, plain-NumPy backward-induction EGM with no dcegm machinery in it --
over a genuine multi-period model with stochastic transitions.

The reference was already general over ``n_periods``; ``grid_for`` now lets its
end-of-period assets grid vary by (period, choice) instead of being one shared
array, which is exactly the reference counterpart of dcegm's
``continuous_grid_functions["assets_end_of_period"]``. Both sides call the *same*
grid function, so they cannot drift apart.

The two tests here divide the work, and neither substitutes for the other:
comparing policy/value against the reference establishes that the *economics* are
right under state-specific grids, but is provably unable to tell which grid was
used (see that test's docstring for the measured numbers); recovering
``endog_grid - policy`` establishes *which grid* was used, but says nothing about
whether the solution on it is correct.

Which variables the grid may depend on is *not* free. Every parent state-choice
transitioning into a shared child must agree on its own grid, enforced by
``check_continuous_grid_consistency_across_shared_children``. In this model
``partner_state`` is stochastic, so both partner states are parents of the same
children -- a grid varying with it is rejected at model-build time (asserted
below). ``period`` and ``choice`` are the safe ones, and are what the main test
varies.

"""

import numpy as np
import pytest

import tests.resources.divorce_model.dcegm_functions as dm
import tests.resources.divorce_model.reference as ref
from tests.test_divorce_toy_model import (
    A_GRID_MAX,
    A_GRID_POINTS,
    N_PERIODS,
    PARAMS,
    dcegm_raw_arrays,
)

# Wider than the plain A_GRID used elsewhere: a married agent's continuation
# wealth is queried at 2*a_end (see the divorce model's budget convention), so the
# reference needs grid coverage up there to interpolate rather than extrapolate.
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
        PARAMS,
        n_periods=N_PERIODS,
        a_grid=BASE_GRID,
        continuous_grid_functions={"assets_end_of_period": assets_grid},
    )
    ref_solved = ref.solve_reference(N_PERIODS, PARAMS, assets_grid)
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
            PARAMS,
            n_periods=N_PERIODS,
            a_grid=BASE_GRID,
            continuous_grid_functions={"assets_end_of_period": bad_grid},
        )
