"""End-to-end tests for state-specific continuous grids on the Druedahl-Jorgensen (n-D
regular) upper envelope path.

Reuses the existing `with_cont_exp` toy model's economic functions unchanged
(utility, budget constraint, state transitions don't care about the upper
envelope method), just switching `upper_envelope["method"]` to
`"druedahl_jorgensen"` and adding the required `assets_begin_of_period` grid. No
dedicated DJ toy model exists in this repo, so this exercises
`interpnd_policy_and_value_for_child_states_on_own_regular_grids` through a real
multi-period backward-induction solve without building one from scratch.

"""

import jax.numpy as jnp
import numpy as np

import dcegm
import dcegm.toy_models as toy_models


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


def _with_none_grid(model_config, name):
    # A continuous_grid_functions entry now requires the matching
    # model_config["continuous_states"] entry to be None.
    config = dict(model_config)
    config["continuous_states"] = dict(model_config["continuous_states"])
    config["continuous_states"][name] = None
    return config


def test_dj_constant_grid_func_reproduces_default_solve_bit_for_bit():
    model_funcs, params, model_specs, model_config = _load_with_cont_exp_dj()

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


def test_dj_constant_but_different_grid_matches_direct_model_config_declaration():
    # The strong correctness check, mirroring the FUES version in
    # test_state_specific_grid_end_to_end.py: a *constant* grid delivered via
    # continuous_grid_functions, but with different values than the model's own
    # default, must reproduce a model where that same grid is declared directly
    # in model_config["continuous_states"] (today's unmodified mechanism)
    # bit-for-bit. Exercises the DJ-specific interpolation axis
    # (interpnd_policy_and_value_for_child_states_on_own_regular_grids) end to end.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp_dj()

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


def test_dj_constant_but_different_grid_matches_direct_declaration_when_simulated():
    # Solve-level equality (test above) doesn't exercise simulation_interp.py's DJ
    # branch, which is a fully separate reader from the interpolation used inside
    # solve. Same bit-for-bit contract, applied to simulate().
    model_funcs, params, model_specs, model_config = _load_with_cont_exp_dj()

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


def test_dj_constant_but_different_grid_matches_direct_declaration_for_choice_queries():
    # Third reader: choice_values_for_states / choice_policies_for_states go
    # through interp_interfaces.py's DJ-multidim branch, a separate code path
    # from both solve and simulate().
    model_funcs, params, model_specs, model_config = _load_with_cont_exp_dj()

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
