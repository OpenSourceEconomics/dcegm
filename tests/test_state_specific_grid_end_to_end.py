"""End-to-end integration tests for state-specific continuous grids.

The unit tests in test_law_of_motion_state_specific_grid.py exercise the grid-
selection mechanism in isolation and would not catch a child-vs-parent mixup
(using the child's own identity to select the grid instead of a representative
parent's) -- that only shows up once real batching/dedup is involved. These tests
instead run the full model-setup + backward-induction pipeline, exercising both
the main backward-induction loop (this toy model has 5 periods) and the
last-two-periods special case in one solve.

"""

import jax.numpy as jnp
import numpy as np
import pytest

import dcegm
import dcegm.toy_models as toy_models


def _load_with_cont_exp():
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    return model_funcs, params, model_specs, model_config


def _with_none_grid(model_config, name):
    # A continuous_grid_functions entry now requires the matching
    # model_config["continuous_states"] entry to be None -- a real array is unused
    # once a grid_func takes over, so it must be declared explicitly absent.
    config = dict(model_config)
    config["continuous_states"] = dict(model_config["continuous_states"])
    config["continuous_states"][name] = None
    return config


def test_constant_grid_func_reproduces_default_solve_bit_for_bit():
    # A grid_func that returns the same grid as the model's default, regardless of
    # state, must reproduce the solve exactly -- this is the critical regression
    # check for Phases 2/4/5: they must be a no-op whenever grids aren't actually
    # state-specific.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    baseline_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    baseline_solved = baseline_model.solve(params)

    default_grid = jnp.asarray(model_config["continuous_states"]["experience"])

    def constant_grid_func(period):
        return default_grid

    state_specific_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "experience"),
        model_specs=model_specs,
        continuous_grid_functions={"experience": constant_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    np.testing.assert_array_equal(
        np.asarray(baseline_solved.value), np.asarray(state_specific_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(baseline_solved.policy), np.asarray(state_specific_solved.policy)
    )
    np.testing.assert_array_equal(
        np.asarray(baseline_solved.endog_grid),
        np.asarray(state_specific_solved.endog_grid),
    )


def test_array_declared_grid_with_grid_function_raises():
    # A continuous_grid_functions entry requires the matching
    # model_config["continuous_states"] entry to be None -- a declared array is
    # unused once a grid_func takes over (only its length mattered, and that's now
    # pinned by evaluating the grid_func against a representative state-choice
    # instead, see continuous_state_grids.py), so leaving a real array there is
    # rejected rather than silently ignored.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()
    default_grid = jnp.asarray(model_config["continuous_states"]["experience"])

    def constant_grid_func(period):
        return default_grid

    with pytest.raises(ValueError, match="is not None"):
        dcegm.setup_model(
            model_config=model_config,
            model_specs=model_specs,
            continuous_grid_functions={"experience": constant_grid_func},
            **model_funcs,
        )


def test_period_dependent_grid_solves_and_differs_from_default():
    # A genuinely period-varying grid -- safe under
    # check_continuous_grid_consistency_across_shared_children since period
    # increments deterministically for every parent of a given child, but the
    # parent's own grid differs from the child's own grid whenever the grid
    # depends on period. This is exactly the scenario that would silently break if
    # the child's identity were (wrongly) used to select the grid instead of a
    # representative parent's: solve output would either crash on a shape
    # mismatch, be non-finite, or (best case, if it happened to run) not reflect
    # the intended per-period scaling correctly.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    default_grid = jnp.asarray(model_config["continuous_states"]["experience"])

    def period_dependent_grid_func(period):
        return default_grid * (1.0 + 0.1 * period)

    varying_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "experience"),
        model_specs=model_specs,
        continuous_grid_functions={"experience": period_dependent_grid_func},
        **model_funcs,
    )
    varying_solved = varying_model.solve(params)
    varying_value = np.asarray(varying_solved.value)

    baseline_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    baseline_solved = baseline_model.solve(params)
    baseline_value = np.asarray(baseline_solved.value)

    # dcegm pads variable-length endogenous grids with NaN up to a common width
    # (confirmed present in the baseline solve too, ~30% of entries here) -- that's
    # an unrelated storage convention, not a sign of a broken solve. What matters:
    # the *set* of valid (non-NaN) entries is unchanged (same states have the same
    # grid lengths regardless of the grid's values), those valid entries are all
    # finite, and they actually differ from the baseline (the feature does
    # something).
    baseline_nan_mask = np.isnan(baseline_value)
    varying_nan_mask = np.isnan(varying_value)
    np.testing.assert_array_equal(baseline_nan_mask, varying_nan_mask)

    assert np.all(np.isfinite(varying_value[~varying_nan_mask]))
    assert not np.allclose(
        varying_value[~varying_nan_mask], baseline_value[~baseline_nan_mask]
    )


def test_constant_but_different_grid_matches_direct_model_config_declaration():
    # The strong correctness check: a *constant* grid delivered via
    # continuous_grid_functions, but with different values than the model's own
    # default, must reproduce a model where that same grid is declared directly in
    # model_config["continuous_states"] (the pre-existing global-grid mechanism)
    # -- bit-for-bit. This is a stronger check than reproducing the *default* grid
    # (test above), because it can only pass if the on-demand grid is actually
    # doing something (using the *wrong* grid, or silently falling back to the
    # default, would both fail this test), and it exercises both the
    # law-of-motion transition input and the interpolation axis / EGM candidate
    # generation at once, since this toy model solves via the FUES (2d irregular)
    # upper envelope by default.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    scaled_grid = jnp.asarray(model_config["continuous_states"]["experience"]) * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["experience"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "experience"),
        model_specs=model_specs,
        continuous_grid_functions={"experience": constant_scaled_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

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


def test_constant_but_different_grid_matches_direct_declaration_when_simulated():
    # The solve-level check above proves the *stored* solution is identical
    # either way, but simulate() reads that solution back out via a completely
    # separate code path (simulation_interp.py), which independently needs to
    # resolve each query's own state-choice-specific grid. A bug there would
    # leave test_constant_but_different_grid_matches_direct_model_config_declaration
    # green (it never simulates) while silently corrupting simulated
    # policy/value/choice output. Same bit-for-bit contract, applied to simulate().
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    scaled_grid = jnp.asarray(model_config["continuous_states"]["experience"]) * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["experience"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "experience"),
        model_specs=model_specs,
        continuous_grid_functions={"experience": constant_scaled_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    n_agents = 1_000
    states_initial = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),
        "experience": np.ones(n_agents) * 0.5,
        "assets_begin_of_period": np.ones(n_agents) * 10,
    }

    df_reference = reference_solved.simulate(states_initial=states_initial, seed=111)
    df_state_specific = state_specific_solved.simulate(
        states_initial=states_initial, seed=111
    )

    for column in df_reference.columns:
        np.testing.assert_array_equal(
            df_reference[column].to_numpy(), df_state_specific[column].to_numpy()
        )


def test_constant_but_different_grid_matches_direct_declaration_for_choice_queries():
    # Third reader path: choice_values_for_states / choice_policies_for_states go
    # through interp_interfaces.py, which is neither the solve path nor
    # simulate()'s simulation_interp.py -- a separate place a leftover
    # shared-grid read could hide.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    scaled_grid = jnp.asarray(model_config["continuous_states"]["experience"]) * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["experience"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "experience"),
        model_specs=model_specs,
        continuous_grid_functions={"experience": constant_scaled_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    n_states = 20
    states = {
        "period": np.zeros(n_states, dtype=int),
        "lagged_choice": np.zeros(n_states, dtype=int),
        "experience": np.linspace(0.0, 1.0, n_states),
        "assets_begin_of_period": np.ones(n_states) * 10,
    }

    reference_values = np.asarray(reference_solved.choice_values_for_states(states))
    state_specific_values = np.asarray(
        state_specific_solved.choice_values_for_states(states)
    )
    np.testing.assert_array_equal(reference_values, state_specific_values)

    reference_policies = np.asarray(reference_solved.choice_policies_for_states(states))
    state_specific_policies = np.asarray(
        state_specific_solved.choice_policies_for_states(states)
    )
    np.testing.assert_array_equal(reference_policies, state_specific_policies)
