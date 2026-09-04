"""End-to-end tests for state-choice-specific continuous grids.

All three grid names go through the same four checks, so they are written once and
parametrized over ``CASES`` rather than duplicated per name and per upper-envelope
method:

``experience``
    An *additional continuous state*. Plays the law-of-motion-input role: a
    representative parent's own grid feeds the child's transition.
``assets_end_of_period``
    The exogenous savings grid. Also a law-of-motion input, and additionally each
    state-choice's own EGM grid.
``assets_begin_of_period``
    Druedahl-Jorgensen's fixed common wealth grid ("m_grid"). Not a law-of-motion
    input at all -- it is the *output* storage grid, so it needs own-grid
    (self-referential) threading rather than a representative parent. Only
    reachable when ``skip_endog_grid_storage`` is True.

The four checks:

1. A grid function returning the model's *own default* must reproduce the plain
   solve bit-for-bit -- the feature must be a no-op when it changes nothing.
2. A constant-but-*different* grid must reproduce a model declaring that same grid
   directly. This is the strong one: it fails both if the wrong grid is used and if
   the grid function is silently ignored.
3. Same, through ``simulate()`` -- a separate reader path from solve.
4. Same, through ``choice_values_for_states``/``choice_policies_for_states`` -- a
   third reader path, distinct from both.

Unit-level config validation lives in ``test_state_specific_grids_config.py``, the
law-of-motion mechanics in ``test_state_specific_grids_law_of_motion.py``, and
checks against independently computed solutions in
``test_state_specific_grids_reference.py``.

"""

import jax.numpy as jnp
import numpy as np
import pytest

import dcegm
import dcegm.toy_models as toy_models
from dcegm.toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    inverse_marginal_utility_crra,
    marginal_utility_crra,
)

N_AGENTS = 1_000
N_QUERY_STATES = 20

# =====================================================================================
# Model loaders
# =====================================================================================


def _with_cont_exp_fues():
    """Continuous experience, FUES upper envelope (the toy model's default)."""
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    return model_funcs, params, model_specs, dict(model_config), "continuous"


def _with_cont_exp_dj():
    """Continuous experience, Druedahl-Jorgensen upper envelope."""
    model_funcs, params, model_specs, model_config, exp_kind = _with_cont_exp_fues()
    model_config["continuous_states"] = dict(model_config["continuous_states"])
    model_config["continuous_states"]["assets_begin_of_period"] = jnp.linspace(
        0, 50, 50
    )
    model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    return model_funcs, params, model_specs, model_config, exp_kind


def _with_exp_dj():
    """Discrete experience -- no additional continuous state -- plus DJ.

    The only configuration in which a state-specific ``assets_begin_of_period`` is
    supported alongside a single-dimensional wealth axis.

    """
    model_funcs = toy_models.load_example_model_functions("with_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_exp")
    )
    model_config = dict(model_config)
    model_config["continuous_states"] = dict(model_config["continuous_states"])
    model_config["continuous_states"]["assets_begin_of_period"] = jnp.linspace(
        0, 50, 50
    )
    model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    return model_funcs, params, model_specs, model_config, "discrete"


# (grid name, loader). Every combination that is actually supported -- see the
# module docstring for why assets_begin_of_period has no FUES entry.
CASES = {
    "experience__fues": ("experience", _with_cont_exp_fues),
    "experience__dj": ("experience", _with_cont_exp_dj),
    "assets_end_of_period__fues": ("assets_end_of_period", _with_cont_exp_fues),
    "assets_end_of_period__dj": ("assets_end_of_period", _with_cont_exp_dj),
    "assets_begin_of_period__dj": ("assets_begin_of_period", _with_exp_dj),
    "assets_begin_of_period__dj_multidim": (
        "assets_begin_of_period",
        _with_cont_exp_dj,
    ),
}


# =====================================================================================
# Shared helpers
# =====================================================================================


def _declared_grid(model_config, name):
    return jnp.asarray(model_config["continuous_states"][name])


def _with_declared_grid(model_config, name, grid):
    config = dict(model_config)
    config["continuous_states"] = dict(model_config["continuous_states"])
    config["continuous_states"][name] = grid
    return config


def _config_for_grid_function(model_config, name):
    """Config to pair with a ``continuous_grid_functions`` entry for ``name``.

    A declared array is unused once a grid function takes over and must be set to
    ``None`` to say so -- except for ``assets_end_of_period``, which is required to
    keep a real array regardless (``check_model_config.py`` reads its length before
    ``continuous_grid_functions`` is even known). See
    ``process_continuous_grid_functions``.

    """
    if name == "assets_end_of_period":
        return model_config
    return _with_declared_grid(model_config, name, None)


def _constant(grid):
    def grid_func(period):
        return grid

    return grid_func


def _initial_states(exp_kind, n):
    experience = np.zeros(n, dtype=int) if exp_kind == "discrete" else np.ones(n) * 0.5
    return {
        "period": np.zeros(n, dtype=int),
        "lagged_choice": np.zeros(n, dtype=int),
        "experience": experience,
        "assets_begin_of_period": np.ones(n) * 10,
    }


def _query_states(exp_kind, n):
    experience = (
        np.arange(n, dtype=int) % 3
        if exp_kind == "discrete"
        else np.linspace(0.0, 1.0, n)
    )
    return {
        "period": np.zeros(n, dtype=int),
        "lagged_choice": np.zeros(n, dtype=int),
        "experience": experience,
        "assets_begin_of_period": np.ones(n) * 10,
    }


_SOLVE_CACHE = {}


def _solve_pair(case, scale):
    """Memoized ``_solve_pair_uncached``.

    Checks 2-4 all interrogate the *same* pair of solved models through three different
    readers, so without this each ``(case, scale)`` would be solved once per check --
    four solves of every model instead of one. Solutions are read-only here, so sharing
    them across tests is safe, and it is what keeps a six-case matrix affordable.

    """
    key = (case, scale)
    if key not in _SOLVE_CACHE:
        _SOLVE_CACHE[key] = _solve_pair_uncached(case, scale)
    return _SOLVE_CACHE[key]


def _solve_pair_uncached(case, scale):
    """Solve the same model twice: grid declared directly vs via a grid function.

    Returns ``(reference_solved, state_specific_solved, exp_kind)``. With ``scale ==
    1.0`` the reference is just the unmodified model, which is check 1; with any other
    scale it is a model declaring the scaled grid directly, which is checks 2-4.

    """
    name, loader = CASES[case]
    model_funcs, params, model_specs, model_config, exp_kind = loader()

    target_grid = _declared_grid(model_config, name) * scale

    reference_config = (
        model_config
        if scale == 1.0
        else _with_declared_grid(model_config, name, target_grid)
    )
    reference_solved = dcegm.setup_model(
        model_config=reference_config, model_specs=model_specs, **model_funcs
    ).solve(params)

    state_specific_solved = dcegm.setup_model(
        model_config=_config_for_grid_function(model_config, name),
        model_specs=model_specs,
        continuous_grid_functions={name: _constant(target_grid)},
        **model_funcs,
    ).solve(params)

    return reference_solved, state_specific_solved, exp_kind


# =====================================================================================
# The four checks, over every supported (grid name, upper envelope) combination
# =====================================================================================


@pytest.mark.parametrize("case", list(CASES))
def test_constant_grid_func_reproduces_default_solve_bit_for_bit(case):
    """A grid function returning the default must change nothing at all."""
    reference_solved, state_specific_solved, _ = _solve_pair(case, scale=1.0)

    np.testing.assert_array_equal(
        np.asarray(reference_solved.value), np.asarray(state_specific_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.policy), np.asarray(state_specific_solved.policy)
    )


@pytest.mark.parametrize("case", list(CASES))
def test_constant_but_different_grid_matches_direct_declaration(case):
    """The strong check: a different grid, delivered two ways, must agree exactly."""
    reference_solved, state_specific_solved, _ = _solve_pair(case, scale=2.0)

    np.testing.assert_array_equal(
        np.asarray(reference_solved.value), np.asarray(state_specific_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.policy), np.asarray(state_specific_solved.policy)
    )


@pytest.mark.parametrize("case", list(CASES))
def test_constant_but_different_grid_matches_direct_declaration_when_simulated(case):
    """Same contract through ``simulate()``, a separate reader from solve.

    A bug confined to the simulation reader would leave the solve-level test above green
    while silently corrupting simulated output.

    """
    reference_solved, state_specific_solved, exp_kind = _solve_pair(case, scale=2.0)
    states_initial = _initial_states(exp_kind, N_AGENTS)

    df_reference = reference_solved.simulate(states_initial=states_initial, seed=111)
    df_state_specific = state_specific_solved.simulate(
        states_initial=states_initial, seed=111
    )

    for column in df_reference.columns:
        np.testing.assert_array_equal(
            df_reference[column].to_numpy(),
            df_state_specific[column].to_numpy(),
            err_msg=f"column {column}",
        )


@pytest.mark.parametrize("case", list(CASES))
def test_constant_but_different_grid_matches_direct_declaration_for_choice_queries(
    case,
):
    """Same contract through the ``choice_*_for_states`` readers, a third path."""
    reference_solved, state_specific_solved, exp_kind = _solve_pair(case, scale=2.0)
    states = _query_states(exp_kind, N_QUERY_STATES)

    np.testing.assert_array_equal(
        np.asarray(reference_solved.choice_values_for_states(states)),
        np.asarray(state_specific_solved.choice_values_for_states(states)),
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.choice_policies_for_states(states)),
        np.asarray(state_specific_solved.choice_policies_for_states(states)),
    )


# =====================================================================================
# Cases that do not fit the matrix
# =====================================================================================


@pytest.mark.parametrize("case", ["experience__fues", "assets_end_of_period__fues"])
def test_period_dependent_grid_solves_and_differs_from_default(case):
    """A genuinely period-varying grid -- the parent/child distinction matters.

    Safe under ``check_continuous_grid_consistency_across_shared_children`` (period
    increments deterministically for every parent of a given child), but the
    parent's own grid differs from the child's. This is exactly what breaks if the
    child's identity is wrongly used to select the grid.

    """
    name, loader = CASES[case]
    model_funcs, params, model_specs, model_config, _ = loader()
    default_grid = _declared_grid(model_config, name)

    def period_dependent_grid_func(period):
        return default_grid * (1.0 + 0.1 * period)

    varying_value = np.asarray(
        dcegm.setup_model(
            model_config=_config_for_grid_function(model_config, name),
            model_specs=model_specs,
            continuous_grid_functions={name: period_dependent_grid_func},
            **model_funcs,
        )
        .solve(params)
        .value
    )
    baseline_value = np.asarray(
        dcegm.setup_model(
            model_config=model_config, model_specs=model_specs, **model_funcs
        )
        .solve(params)
        .value
    )

    # dcegm pads variable-length endogenous grids with NaN to a common width; that
    # padding pattern is a storage convention and must be unchanged, while the
    # valid entries must be finite and actually different from the baseline.
    baseline_nan = np.isnan(baseline_value)
    varying_nan = np.isnan(varying_value)
    np.testing.assert_array_equal(baseline_nan, varying_nan)
    assert np.all(np.isfinite(varying_value[~varying_nan]))
    assert not np.allclose(varying_value[~varying_nan], baseline_value[~baseline_nan])


def test_state_specific_grid_by_group_matches_separate_per_group_models():
    """Ground truth: one model with a per-group grid == two single-group models.

    ``group`` enters neither utility, budget, nor law of motion, so the two groups'
    economics are identical except for which grid their solution is stored and
    interpolated on. A combined solve must therefore reproduce each separate solve
    exactly -- an equivalence that holds only if the per-group grid is threaded
    correctly all the way through.

    """
    model_funcs, params, model_specs, model_config, _ = _with_cont_exp_fues()

    default_grid = _declared_grid(model_config, "experience")
    group1_grid = default_grid * 2.0

    def grid_func(group):
        return jnp.where(group == 0, default_grid, group1_grid)

    grouped_config = _with_declared_grid(model_config, "experience", None)
    grouped_config["deterministic_states"] = {"group": [0, 1]}
    grouped_solved = dcegm.setup_model(
        model_config=grouped_config,
        model_specs=model_specs,
        continuous_grid_functions={"experience": grid_func},
        **model_funcs,
    ).solve(params)

    solved_per_group = [
        dcegm.setup_model(
            model_config=_with_declared_grid(model_config, "experience", grid),
            model_specs=model_specs,
            **model_funcs,
        ).solve(params)
        for grid in (default_grid, group1_grid)
    ]

    base_states = _query_states("continuous", 15)
    for group, solved_separately in enumerate(solved_per_group):
        states = {**base_states, "group": np.full(15, group, dtype=int)}
        np.testing.assert_array_equal(
            np.asarray(grouped_solved.choice_values_for_states(states)),
            np.asarray(solved_separately.choice_values_for_states(base_states)),
        )
        np.testing.assert_array_equal(
            np.asarray(grouped_solved.choice_policies_for_states(states)),
            np.asarray(solved_separately.choice_policies_for_states(base_states)),
        )


def _utility_crra_with_experience(consumption, choice, experience, params):
    """CRRA utility that depends on the continuous state *directly*.

    ``with_cont_exp``'s own utility takes only ``consumption`` and ``choice``, so
    ``determine_function_arguments_and_partial_model_specs`` filters "experience"
    out before it reaches the function -- meaning a wrong experience value would
    have no effect on the output at all. This restores that dependence.

    """
    import jax

    utility_consumption = jax.lax.select(
        jnp.allclose(params["rho"], 1),
        jnp.log(consumption),
        (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"]),
    )
    return (
        utility_consumption
        - (1 - choice) * params["delta"]
        + params["exp"] * experience
    )


def test_upper_envelope_refinement_uses_each_state_choices_own_grid():
    """Regression test for the upper-envelope step reading the shared grid.

    ``run_upper_envelope`` refines the EGM candidates *after* they are generated,
    and used to feed the model-wide default grid into the upper envelope's
    ``value_function`` regardless of any per-state-choice override.

    Two things are needed to make that observable, which is why this cannot be
    folded into the matrix above: a utility function that depends on "experience"
    directly (see above), and the Druedahl-Jorgensen upper envelope, which
    evaluates ``value_function`` unconditionally at every point. FUES only calls it
    when a credit-constrained non-monotonicity is present, which this
    parameterization does not trigger.

    """
    model_funcs, params, model_specs, model_config, _ = _with_cont_exp_dj()
    model_funcs = dict(model_funcs)
    model_funcs["utility_functions"] = {
        "utility": _utility_crra_with_experience,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
    }

    scaled_grid = _declared_grid(model_config, "experience") * 2.0

    reference_solved = dcegm.setup_model(
        model_config=_with_declared_grid(model_config, "experience", scaled_grid),
        model_specs=model_specs,
        **model_funcs,
    ).solve(params)
    state_specific_solved = dcegm.setup_model(
        model_config=_with_declared_grid(model_config, "experience", None),
        model_specs=model_specs,
        continuous_grid_functions={"experience": _constant(scaled_grid)},
        **model_funcs,
    ).solve(params)

    np.testing.assert_array_equal(
        np.asarray(reference_solved.value), np.asarray(state_specific_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.policy), np.asarray(state_specific_solved.policy)
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.endog_grid),
        np.asarray(state_specific_solved.endog_grid),
    )
