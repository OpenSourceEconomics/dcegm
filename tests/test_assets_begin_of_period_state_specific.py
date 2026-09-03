"""End-to-end tests for state-choice-specific `assets_begin_of_period` grids.

Unlike `assets_end_of_period`, `assets_begin_of_period` is not a law-of-motion
input -- it's the *output* common wealth grid ("m_grid") Druedahl-Jorgensen
interpolates onto and stores against, so it needs "own grid" (self-referential)
threading, not representative-parent threading: reading a state-choice's own
stored solution back needs no parent, just that state-choice's own identity.

Only reachable at all when `skip_endog_grid_storage` is True
(`upper_envelope["method"] == "druedahl_jorgensen"` and >= 2 choices) -- that's
exactly the case where `endog_grid` isn't stored, because every state-choice's
"endogenous" grid is by construction this fixed array. It also does not (yet)
support the n-D regular (multiple additional continuous states) interpolation
path, which still treats the wealth grid as shared across children -- see
process_continuous_grid_functions's validation. These tests use `with_cont_exp`'s
plain wealth-only Druedahl-Jorgensen configuration (no additional continuous
state) to stay inside supported territory.

"""

import jax.numpy as jnp
import numpy as np

import dcegm
import dcegm.toy_models as toy_models


def _load_with_exp_dj_no_additional_continuous_state():
    # "with_exp" (discrete experience, no additional continuous state) rather than
    # "with_cont_exp" -- assets_begin_of_period state-specificity is only supported
    # without additional continuous states.
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
    return model_funcs, params, model_specs, model_config


def _with_none_grid(model_config, name):
    # A continuous_grid_functions entry now requires the matching
    # model_config["continuous_states"] entry to be None.
    config = dict(model_config)
    config["continuous_states"] = dict(model_config["continuous_states"])
    config["continuous_states"][name] = None
    return config


def test_constant_grid_func_reproduces_default_solve_bit_for_bit():
    model_funcs, params, model_specs, model_config = (
        _load_with_exp_dj_no_additional_continuous_state()
    )

    baseline_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    baseline_solved = baseline_model.solve(params)

    default_grid = jnp.asarray(
        model_config["continuous_states"]["assets_begin_of_period"]
    )

    def constant_grid_func(period):
        return default_grid

    state_specific_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "assets_begin_of_period"),
        model_specs=model_specs,
        continuous_grid_functions={"assets_begin_of_period": constant_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    np.testing.assert_array_equal(
        np.asarray(baseline_solved.value), np.asarray(state_specific_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(baseline_solved.policy), np.asarray(state_specific_solved.policy)
    )


def test_constant_but_different_grid_matches_direct_declaration_solve():
    # The strong correctness check: a constant grid delivered via
    # continuous_grid_functions, but with different values than the model's own
    # declared array, must reproduce a model where that same grid is declared
    # directly -- bit-for-bit.
    model_funcs, params, model_specs, model_config = (
        _load_with_exp_dj_no_additional_continuous_state()
    )

    default_grid = jnp.asarray(
        model_config["continuous_states"]["assets_begin_of_period"]
    )
    scaled_grid = default_grid * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["assets_begin_of_period"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "assets_begin_of_period"),
        model_specs=model_specs,
        continuous_grid_functions={"assets_begin_of_period": constant_scaled_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    np.testing.assert_array_equal(
        np.asarray(reference_solved.value), np.asarray(state_specific_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.policy), np.asarray(state_specific_solved.policy)
    )


def test_constant_but_different_grid_matches_direct_declaration_when_simulated():
    # Same strong check, through simulate() -- a separate reader path from solve.
    model_funcs, params, model_specs, model_config = (
        _load_with_exp_dj_no_additional_continuous_state()
    )

    default_grid = jnp.asarray(
        model_config["continuous_states"]["assets_begin_of_period"]
    )
    scaled_grid = default_grid * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["assets_begin_of_period"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "assets_begin_of_period"),
        model_specs=model_specs,
        continuous_grid_functions={"assets_begin_of_period": constant_scaled_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    n_agents = 1_000
    states_initial = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),
        "experience": np.zeros(n_agents, dtype=int),
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
    # Same strong check, through choice_values_for_states/choice_policies_for_states
    # -- a third, separate reader path from both solve and simulate().
    model_funcs, params, model_specs, model_config = (
        _load_with_exp_dj_no_additional_continuous_state()
    )

    default_grid = jnp.asarray(
        model_config["continuous_states"]["assets_begin_of_period"]
    )
    scaled_grid = default_grid * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["assets_begin_of_period"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "assets_begin_of_period"),
        model_specs=model_specs,
        continuous_grid_functions={"assets_begin_of_period": constant_scaled_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    n_states = 5
    states = {
        "period": np.zeros(n_states, dtype=int),
        "lagged_choice": np.zeros(n_states, dtype=int),
        "experience": np.arange(n_states, dtype=int),
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


def _load_with_cont_exp_dj():
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    model_config = dict(model_config)
    model_config["continuous_states"] = dict(model_config["continuous_states"])
    model_config["continuous_states"]["assets_begin_of_period"] = jnp.linspace(
        0, 50, 50
    )
    model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    return model_funcs, params, model_specs, model_config


def test_multidim_continuous_state_constant_but_different_grid_matches_direct_declaration_solve():
    # with_cont_exp has two choices, so skip_endog_grid_storage is True (see
    # check_model_config.py) and a state-specific assets_begin_of_period is
    # allowed alongside "experience" (an additional continuous state, left at its
    # real default grid here) -- interpnd_regular.py's own-grid path now threads
    # wealth_grid per child, mirroring additional_continuous_state_grids_per_child.
    # Same strong bit-for-bit correctness check as the no-additional-continuous-
    # state tests above.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp_dj()

    default_grid = jnp.asarray(
        model_config["continuous_states"]["assets_begin_of_period"]
    )
    scaled_grid = default_grid * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["assets_begin_of_period"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model = dcegm.setup_model(
        model_config=_with_none_grid(model_config, "assets_begin_of_period"),
        model_specs=model_specs,
        continuous_grid_functions={"assets_begin_of_period": constant_scaled_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    np.testing.assert_array_equal(
        np.asarray(reference_solved.value), np.asarray(state_specific_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.policy), np.asarray(state_specific_solved.policy)
    )


def test_multidim_continuous_state_single_choice_rejects_state_specific_assets_begin_of_period():
    # With a single choice, skip_endog_grid_storage is False: the Druedahl-
    # Jorgensen upper envelope is skipped entirely, so the stored endogenous grid
    # is not the fixed assets_begin_of_period grid there, and a state-specific
    # assets_begin_of_period has nothing to plug into (unlike the two-choice case
    # exercised above). Unit-level version in test_state_specific_continuous_grids.py;
    # this confirms it also surfaces through the full dcegm.setup_model entry point.
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    model_config = dict(model_config)
    model_config["choices"] = [0]
    model_config["continuous_states"] = dict(model_config["continuous_states"])
    model_config["continuous_states"]["assets_begin_of_period"] = None
    model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}

    def constant_grid_func(period):
        return jnp.linspace(0, 50, 50)

    import pytest

    with pytest.raises(ValueError, match="skip_endog_grid_storage"):
        dcegm.setup_model(
            model_config=model_config,
            model_specs=model_specs,
            continuous_grid_functions={"assets_begin_of_period": constant_grid_func},
            **model_funcs,
        )
