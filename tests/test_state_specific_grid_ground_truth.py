"""Ground-truth validation for state-specific continuous grids.

Solves the same economics two ways and checks they agree bit-for-bit: once as one
model with a genuine discrete "type" state (`group`, via `deterministic_states` --
a built-in `dcegm` mechanism whose default law of motion already leaves it
unchanged across periods, so it structurally satisfies the consistency check in
continuous_state_grids.py) whose `continuous_grid_functions` grid depends on that
type; once as two fully separate single-type models, each declaring its own
type's grid directly via `model_config["continuous_states"]` (the pre-existing
mechanism). `group` doesn't enter `with_cont_exp`'s utility, budget, or
law-of-motion functions at all, so the two groups' economics are identical except
for which grid the solution is stored/interpolated on -- meaning a combined solve
must reproduce the two separate solves exactly, for each group.

The negative case (a grid depending on a variable that does not survive the
parent->child transition 1:1 must be rejected at build time) is covered by
test_grid_depending_on_lagged_choice_fails and
test_grid_on_state_that_does_not_pass_through_1to1_fails in
test_state_specific_continuous_grids.py, using the same "group" mechanism as here.

"""

import jax.numpy as jnp
import numpy as np

import dcegm
import dcegm.toy_models as toy_models


def test_state_specific_grid_by_group_matches_separate_per_group_models():
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )

    default_grid = jnp.asarray(model_config["continuous_states"]["experience"])
    group1_grid = default_grid * 2.0

    def grid_func(group):
        return jnp.where(group == 0, default_grid, group1_grid)

    grouped_config = dict(model_config)
    grouped_config["deterministic_states"] = {"group": [0, 1]}
    grouped_config["continuous_states"] = dict(model_config["continuous_states"])
    grouped_config["continuous_states"]["experience"] = None
    grouped_model = dcegm.setup_model(
        model_config=grouped_config,
        model_specs=model_specs,
        continuous_grid_functions={"experience": grid_func},
        **model_funcs,
    )
    grouped_solved = grouped_model.solve(params)

    # Reference: two fully separate single-group models (no "group" state at all),
    # each declaring its group's grid directly, the pre-existing mechanism.
    ref_config_g0 = dict(model_config)
    model_g0 = dcegm.setup_model(
        model_config=ref_config_g0, model_specs=model_specs, **model_funcs
    )
    solved_g0 = model_g0.solve(params)

    ref_config_g1 = dict(model_config)
    ref_config_g1["continuous_states"] = dict(model_config["continuous_states"])
    ref_config_g1["continuous_states"]["experience"] = group1_grid
    model_g1 = dcegm.setup_model(
        model_config=ref_config_g1, model_specs=model_specs, **model_funcs
    )
    solved_g1 = model_g1.solve(params)

    n_states = 15
    base_states = {
        "period": np.zeros(n_states, dtype=int),
        "lagged_choice": np.zeros(n_states, dtype=int),
        "experience": np.linspace(0.0, 1.0, n_states),
        "assets_begin_of_period": np.ones(n_states) * 10,
    }
    states_g0 = {**base_states, "group": np.zeros(n_states, dtype=int)}
    states_g1 = {**base_states, "group": np.ones(n_states, dtype=int)}

    np.testing.assert_array_equal(
        np.asarray(grouped_solved.choice_values_for_states(states_g0)),
        np.asarray(solved_g0.choice_values_for_states(base_states)),
    )
    np.testing.assert_array_equal(
        np.asarray(grouped_solved.choice_values_for_states(states_g1)),
        np.asarray(solved_g1.choice_values_for_states(base_states)),
    )
    np.testing.assert_array_equal(
        np.asarray(grouped_solved.choice_policies_for_states(states_g0)),
        np.asarray(solved_g0.choice_policies_for_states(base_states)),
    )
    np.testing.assert_array_equal(
        np.asarray(grouped_solved.choice_policies_for_states(states_g1)),
        np.asarray(solved_g1.choice_policies_for_states(base_states)),
    )
