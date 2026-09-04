"""End-to-end tests for state-choice-specific `assets_end_of_period` grids.

`assets_end_of_period` is a law-of-motion input (the exogenous savings grid a
parent's transition is evaluated over, feeding the child's own budget equation) --
the same role the *additional* continuous states (e.g. "experience") already have,
so it needs the same representative-parent threading through
`calc_law_of_motion_for_state_choices`. It is also its own state-choice's own
combo/wealth axis when that state-choice is solving its own EGM problem
(`solve_euler_equation.py`) or its own terminal value
(`final_periods.py`'s FUES branch) -- self-referential there, no representative
parent needed, mirroring the additional continuous states' "own grid" role in the
same functions.

Unlike the additional continuous states and `assets_begin_of_period`,
`assets_end_of_period` does not support the `None`-grid convention: its declared
array fixes not just a length but the FUES tuning-parameter defaults
(`n_constrained_points_to_add`, `extra_wealth_grid_factor`-derived
`n_total_wealth_grid`), computed long before continuous_grid_functions is even
processed. A real array must always be declared in
`model_config["continuous_states"]["assets_end_of_period"]`; a
continuous_grid_functions entry may still override its *values* state-choice-
specifically, as long as every state-choice's own grid has that same declared
length.

"""

import jax.numpy as jnp
import numpy as np

import dcegm
import dcegm.toy_models as toy_models


def _load_with_cont_exp():
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    return model_funcs, params, model_specs, model_config


def _load_with_cont_exp_dj():
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()
    model_config = dict(model_config)
    model_config["continuous_states"] = dict(model_config["continuous_states"])
    model_config["continuous_states"]["assets_begin_of_period"] = jnp.linspace(
        0, 50, 50
    )
    model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    return model_funcs, params, model_specs, model_config


def test_constant_grid_func_reproduces_default_solve_bit_for_bit():
    # A grid_func that returns the model's own declared array, regardless of state,
    # must reproduce the solve exactly -- the critical regression check that the
    # representative-parent threading is a no-op whenever the grid isn't actually
    # state-specific.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    baseline_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    baseline_solved = baseline_model.solve(params)

    default_grid = jnp.asarray(
        model_config["continuous_states"]["assets_end_of_period"]
    )

    def constant_grid_func(period):
        return default_grid

    state_specific_model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        continuous_grid_functions={"assets_end_of_period": constant_grid_func},
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


def test_constant_but_different_grid_matches_direct_model_config_declaration():
    # The strong correctness check: a *constant* grid delivered via
    # continuous_grid_functions, but with different values (same length, so tuning
    # params stay identical) than the model's own declared array, must reproduce a
    # model where that same grid is declared directly -- bit-for-bit. Exercises both
    # the law-of-motion transition-input role (a representative parent's own grid
    # feeding a child's budget equation) and the own-grid role (this state-choice's
    # own EGM candidate generation) at once.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    default_grid = jnp.asarray(
        model_config["continuous_states"]["assets_end_of_period"]
    )
    scaled_grid = default_grid * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["assets_end_of_period"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        continuous_grid_functions={"assets_end_of_period": constant_scaled_grid_func},
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


def test_period_dependent_grid_solves_and_differs_from_default():
    # A genuinely period-varying grid -- safe under the child-sharing consistency
    # check since period increments deterministically for every parent of a given
    # child, but the parent's own grid differs from the child's own grid whenever
    # the grid depends on period. Exactly the scenario that would silently break if
    # the child's identity were used to select the grid instead of a representative
    # parent's.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    default_grid = jnp.asarray(
        model_config["continuous_states"]["assets_end_of_period"]
    )

    def period_dependent_grid_func(period):
        return default_grid * (1.0 + 0.1 * period)

    varying_model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        continuous_grid_functions={"assets_end_of_period": period_dependent_grid_func},
        **model_funcs,
    )
    varying_solved = varying_model.solve(params)
    varying_value = np.asarray(varying_solved.value)

    baseline_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    baseline_solved = baseline_model.solve(params)
    baseline_value = np.asarray(baseline_solved.value)

    baseline_nan_mask = np.isnan(baseline_value)
    varying_nan_mask = np.isnan(varying_value)
    np.testing.assert_array_equal(baseline_nan_mask, varying_nan_mask)

    assert np.all(np.isfinite(varying_value[~varying_nan_mask]))
    assert not np.allclose(
        varying_value[~varying_nan_mask], baseline_value[~baseline_nan_mask]
    )


def test_dj_constant_but_different_grid_matches_direct_model_config_declaration():
    # Same strong correctness check as the FUES version above, but through the
    # Druedahl-Jorgensen path -- law_of_motion.py's representative-parent threading
    # and solve_euler_equation.py's own-grid role are both upper-envelope-method
    # agnostic, so this must hold here too.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp_dj()

    default_grid = jnp.asarray(
        model_config["continuous_states"]["assets_end_of_period"]
    )
    scaled_grid = default_grid * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["assets_end_of_period"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        continuous_grid_functions={"assets_end_of_period": constant_scaled_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    np.testing.assert_array_equal(
        np.asarray(reference_solved.value), np.asarray(state_specific_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.policy), np.asarray(state_specific_solved.policy)
    )
