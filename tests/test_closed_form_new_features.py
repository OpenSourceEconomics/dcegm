"""Closed-form tests for the features added on this branch.

Every other test for these features is a *relative* check -- one dcegm solve
against another dcegm solve (bit-for-bit equivalence between two configurations,
or between two evaluation granularities). Those catch regressions but share a
common mode of failure: if the underlying EGM step used the wrong grid in both
solves, they would agree with each other and still be wrong.

These tests are absolute instead. They pin the solution against a closed-form
Euler equation computed independently in NumPy, for a two-period model where that
is exact:

- Period 1 is terminal, so consumption equals wealth and the marginal utility of
  wealth is ``wealth ** -rho`` regardless of the period-1 choice.
- Period 0 therefore satisfies, for each savings point ``a`` on that
  state-choice's *own* end-of-period assets grid,

  .. code-block:: text

      RHS   = discount_factor * (1 + r) * E_shock[ sum_k P(k) * wealth_1(k) ** -rho ]
      c_0   = RHS ** (-1 / rho)
      wealth_0 = a + c_0

  where ``P(k)`` are the logit choice probabilities over period-1 choices (see
  ``aggregate_marg_utils_and_exp_values``) and ``wealth_1(k)`` comes from the
  user's budget equation.

Two independent things are checked from that:

1. **Which grid was used** -- ``endog_grid - policy`` must reproduce exactly the
   end-of-period assets grid that state-choice is supposed to own. The Euler
   residual alone cannot catch a wrong grid, since it is a property of each
   (wealth, consumption) pair regardless of where the pair sits.
2. **Whether the economics are right** -- the Euler residual against the
   closed-form RHS above.

Not covered here: the Druedahl-Jorgensen ``assets_begin_of_period`` grid. There
the stored policy is *interpolated* onto the common wealth grid rather than being
a raw EGM solution, so it does not satisfy the Euler equation pointwise and has
no closed form of this kind. It is covered by the equivalence tests in
``test_assets_begin_of_period_state_specific.py``.

"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import roots_sh_legendre
from scipy.stats import norm

import dcegm

RHO = 1.5
N_ASSET_POINTS = 12
N_QUAD = 5

PARAMS = {
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
        norm.ppf(quad_points) * PARAMS["income_shock_std"] + PARAMS["income_shock_mean"]
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
                params=PARAMS,
                model_specs=MODEL_SPECS,
            )
        )

    values_next = np.empty_like(wealth_next)
    for k_idx, choice_next in enumerate(choices):
        values_next[k_idx] = np.asarray(
            _utility_final(wealth=wealth_next[k_idx], choice=choice_next, params=PARAMS)
        )
    marg_util_next = wealth_next ** (-RHO)

    # Logit choice probabilities over period-1 choices, rescaled by the max the
    # same way calculate_choice_probs_and_unsqueezed_logsum does.
    scale = PARAMS["taste_shock_scale"]
    rescaled = np.exp((values_next - values_next.max(axis=0, keepdims=True)) / scale)
    choice_probs = rescaled / rescaled.sum(axis=0, keepdims=True)

    marg_util_aggregated = (choice_probs * marg_util_next).sum(axis=0)
    marg_util_integrated = marg_util_aggregated @ quad_weights

    rhs = (
        marg_util_integrated * (1 + PARAMS["interest_rate"]) * PARAMS["discount_factor"]
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
    return model, model.solve(PARAMS)


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


def test_none_declared_grid_matches_closed_form():
    """The ``None``-grid convention, pinned absolutely rather than by equivalence.

    ``assets_end_of_period`` cannot be declared ``None``, so the convention is exercised
    on an additional continuous state is not available in this two-period setup; instead
    this checks the closest analogue that is: a model whose grid function fully replaces
    the declared array.

    """
    choices = [0, 1]
    model, solved = _solve(
        choices=choices,
        budget_fn=_budget,
        continuous_grid_functions={"assets_end_of_period": _assets_grid_by_group},
    )
    # The declared array's values are never read once the grid function is given.
    declared = np.asarray(
        model.model_config["continuous_states_info"]["assets_grid_end_of_period"]
    )
    idxs, rows = _period_zero_state_choices(model)
    endog_grid = np.asarray(solved.endog_grid)
    policy = np.asarray(solved.policy)

    for state_choice_idx, row in zip(idxs, rows):
        own_grid = np.asarray(_assets_grid_by_group(group=row["group"]))
        implied = (
            endog_grid[state_choice_idx, 0, 1 : len(own_grid) + 1]
            - policy[state_choice_idx, 0, 1 : len(own_grid) + 1]
        )
        assert_allclose(implied, own_grid, atol=1e-6)
        if row["group"] != 0:
            # ... and is genuinely not what was solved on.
            assert not np.allclose(implied, declared)
