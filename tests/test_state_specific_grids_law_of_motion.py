"""Law-of-motion tests for the grid/granularity behaviour added on this branch.

Two closely related concerns live here, both unit-level tests of
``law_of_motion.py``:

1. **Whose grid feeds the transition** -- the representative-parent selection.
   A child's transition is evaluated over the *parent's* own continuous grid,
   not the child's, which only matters once grids are state-choice-specific.

2. **At what granularity it is evaluated** -- the state-level dedup. The
   transition into a child does not depend on the child's own future choice, so
   unless a user transition function declares ``choice``,
   ``calc_law_of_motion`` evaluates once per unique child *state* and gathers the
   result out instead of recomputing it per state-choice. That is purely a cost
   optimization, so the central test is an equivalence check: forcing the
   per-state-choice path by adding an otherwise-unused ``choice`` argument to the
   budget equation must reproduce the deduplicated path bit-for-bit.

End-to-end consequences of state-specific grids are in
``test_state_specific_grids_end_to_end.py``; the config layer is in
``test_state_specific_grids_config.py``.

"""

import jax.numpy as jnp
import numpy as np
from jax import vmap

import dcegm
import dcegm.toy_models as toy_models
from dcegm.law_of_motion import (
    _continuous_state_next_period_for_one_state,
    _get_continuous_state_next_period,
    compute_own_continuous_grid_combos,
)
from dcegm.toy_models.cons_ret_model_with_cont_exp.budget_constraint import (
    budget_constraint_cont_exp,
)

# =====================================================================================
# Representative-parent grid selection
# =====================================================================================


def test_compute_own_continuous_grid_combos_single_continuous_state():
    state_dict = {"group": jnp.array(1)}

    def grid_func(group):
        return jnp.where(group == 0, jnp.array([0.0, 1.0]), jnp.array([10.0, 11.0]))

    combos = compute_own_continuous_grid_combos(
        state_dict=state_dict,
        continuous_grid_functions={"experience": grid_func},
        additional_continuous_state_names=["experience"],
    )
    np.testing.assert_allclose(combos["experience"], jnp.array([10.0, 11.0]))


def test_compute_own_continuous_grid_combos_meshes_multiple_names():
    state_dict = {"group": jnp.array(0)}
    grid_functions = {
        "a": lambda group: jnp.array([0.0, 1.0]),
        "b": lambda group: jnp.array([10.0, 20.0, 30.0]),
    }

    combos = compute_own_continuous_grid_combos(
        state_dict=state_dict,
        continuous_grid_functions=grid_functions,
        additional_continuous_state_names=["a", "b"],
    )
    expected_a, expected_b = jnp.meshgrid(
        jnp.array([0.0, 1.0]), jnp.array([10.0, 20.0, 30.0]), indexing="ij"
    )
    np.testing.assert_allclose(combos["a"], expected_a.ravel())
    np.testing.assert_allclose(combos["b"], expected_b.ravel())


def test_continuous_state_next_period_for_one_state_uses_each_states_own_grid():
    # Two states (group 0 and group 1), each with its own 2-point "experience" grid.
    # Exercises the actual production function (vmapped exactly as
    # _get_continuous_state_next_period does), not a standalone reference
    # implementation.
    state_space_dict = {"group": jnp.array([0, 1])}

    def grid_func(group):
        return jnp.where(group == 0, jnp.array([0.0, 1.0]), jnp.array([10.0, 11.0]))

    def compute_continuous_state(group, experience, params):
        return {"experience": experience + 100.0 * group}

    result = vmap(
        _continuous_state_next_period_for_one_state,
        in_axes=(0, 0, None, None, None, None),
    )(
        state_space_dict,
        state_space_dict,  # representative_parent_state_choice_vec == state_space_dict: no parent/child distinction being tested here, see the dedicated test for that below.
        {"experience": grid_func},
        ["experience"],
        {},
        compute_continuous_state,
    )

    # State 0 (group=0) transitions from its own grid [0, 1].
    np.testing.assert_allclose(result["experience"][0], jnp.array([0.0, 1.0]))
    # State 1 (group=1) transitions from its own, different grid [10, 11] -- if the
    # grid were wrongly shared/broadcast, this would come out as [10, 11] + 100, not
    # distinguishable from state 0 having (wrongly) used the same grid.
    np.testing.assert_allclose(result["experience"][1], jnp.array([110.0, 111.0]))


def test_continuous_state_next_period_for_one_state_constant_grid_matches_outer_product():
    # Regression check: when every state's grid_func returns the same grid
    # (today's global-grid behavior), the per-state fused computation must
    # reproduce exactly the outer-product result the old broadcast vmap produced.
    state_space_dict = {"group": jnp.array([0, 1, 2])}
    shared_grid = jnp.array([0.0, 1.0, 2.0, 3.0])

    def grid_func(group):
        return shared_grid

    def compute_continuous_state(group, experience, params):
        return {"experience": experience + group}

    result = vmap(
        _continuous_state_next_period_for_one_state,
        in_axes=(0, 0, None, None, None, None),
    )(
        state_space_dict,
        state_space_dict,
        {"experience": grid_func},
        ["experience"],
        {},
        compute_continuous_state,
    )

    expected = shared_grid[None, :] + jnp.array([0, 1, 2])[:, None]
    np.testing.assert_allclose(result["experience"], expected)


def test_get_continuous_state_next_period_dummy_path_unaffected():
    # Models without an additional continuous state never touch
    # continuous_grid_functions at all.
    state_space_dict = {"group": jnp.array([0, 1])}

    result = _get_continuous_state_next_period(
        has_additional_continuous_states=False,
        child_state_choices=state_space_dict,
        representative_last_period_parent_states=state_space_dict,
        additional_continuous_state_names=[],
        params={},
        model_funcs={},
    )
    assert result["dummy_cont"].shape == (2, 1)


def test_get_continuous_state_next_period_uses_representative_parent_not_state_space_dict():
    # state_space_dict is the CHILD's own identity (used for the transition function
    # call itself); representative_parent_state_choice_vec is a representative PARENT's
    # identity (used only to pick which grid to feed in). These must be allowed to
    # differ -- this is the exact bug this test guards against: using the child's
    # identity to select the grid instead of the parent's.
    state_space_dict = {"group": jnp.array([9, 9])}  # child's own group -- irrelevant
    representative_parent_state_choice_vec = {
        "group": jnp.array([0, 1])
    }  # representative parent's group

    def grid_func(group):
        return jnp.where(group == 0, jnp.array([0.0, 1.0]), jnp.array([10.0, 11.0]))

    def compute_continuous_state(group, experience, params):
        return {"experience": experience}

    result = _get_continuous_state_next_period(
        has_additional_continuous_states=True,
        child_state_choices=state_space_dict,
        representative_last_period_parent_states=representative_parent_state_choice_vec,
        additional_continuous_state_names=["experience"],
        params={},
        model_funcs={
            "continuous_grid_functions": {"experience": grid_func},
            "next_period_continuous_state": compute_continuous_state,
        },
    )
    # If the child's identity (group=9 for both rows) had wrongly been used for grid
    # selection, grid_func(9) would give [10, 11] for both rows.
    np.testing.assert_allclose(result["experience"][0], jnp.array([0.0, 1.0]))
    np.testing.assert_allclose(result["experience"][1], jnp.array([10.0, 11.0]))


# =====================================================================================
# Evaluation granularity: state-level dedup vs per state-choice
# =====================================================================================


def _load_with_cont_exp():
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    return model_funcs, params, model_specs, model_config


def _budget_constraint_with_unused_choice(
    period,
    lagged_choice,
    choice,
    experience,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
    model_specs,
):
    """Identical economics to budget_constraint_cont_exp, but declaring ``choice``.

    ``choice`` is deliberately unused in the body: it only changes which granularity the
    law of motion is evaluated at (per state-choice rather than per unique child state),
    which must not change the result.

    """
    return budget_constraint_cont_exp(
        period=period,
        lagged_choice=lagged_choice,
        experience=experience,
        asset_end_of_previous_period=asset_end_of_previous_period,
        income_shock_previous_period=income_shock_previous_period,
        params=params,
        model_specs=model_specs,
    )


def test_default_toy_models_take_the_state_level_fast_path():
    # None of the shipped toy models' transition functions declare "choice", so
    # they must all take the deduplicated path -- otherwise the rest of the test
    # suite would never exercise it.
    for name in ["with_cont_exp", "with_exp", "dcegm_paper"]:
        model_funcs = toy_models.load_example_model_functions(name)
        params, model_specs, model_config = (
            toy_models.load_example_params_model_specs_and_config(name)
        )
        model = dcegm.setup_model(
            model_config=model_config, model_specs=model_specs, **model_funcs
        )
        assert not model.model_funcs["transition_funcs_depend_on_choice"]["any"], name


def test_choice_in_budget_signature_selects_the_state_choice_path():
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()
    model_funcs = dict(model_funcs)
    model_funcs["budget_constraint"] = _budget_constraint_with_unused_choice

    model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    assert model.model_funcs["transition_funcs_depend_on_choice"]["any"]


def test_state_level_and_state_choice_level_paths_agree_bit_for_bit():
    # The core equivalence check: a budget equation that declares (but ignores)
    # "choice" forces the per-state-choice path, and must reproduce the
    # deduplicated per-state path exactly. A dedup/gather bug -- e.g. a wrong
    # state_row_for_state_choice mapping -- would show up here as mismatched
    # value/policy, since it would feed each child the wrong state's transition.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    fast_path_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    assert not fast_path_model.model_funcs["transition_funcs_depend_on_choice"]["any"]
    fast_path_solved = fast_path_model.solve(params)

    slow_path_funcs = dict(model_funcs)
    slow_path_funcs["budget_constraint"] = _budget_constraint_with_unused_choice
    slow_path_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **slow_path_funcs
    )
    assert slow_path_model.model_funcs["transition_funcs_depend_on_choice"]["any"]
    slow_path_solved = slow_path_model.solve(params)

    np.testing.assert_array_almost_equal(
        np.asarray(fast_path_solved.value), np.asarray(slow_path_solved.value)
    )
    np.testing.assert_array_almost_equal(
        np.asarray(fast_path_solved.policy), np.asarray(slow_path_solved.policy)
    )
    np.testing.assert_array_almost_equal(
        np.asarray(fast_path_solved.endog_grid),
        np.asarray(slow_path_solved.endog_grid),
    )


def test_genuinely_choice_dependent_budget_differs_from_choice_free_one():
    # Sensitivity check for the test above: confirms a budget equation that
    # actually *uses* choice produces a different solution, so the bit-for-bit
    # agreement there reflects the two paths genuinely computing the same thing,
    # not "choice" being unable to affect the budget at all.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    def budget_constraint_using_choice(
        period,
        lagged_choice,
        choice,
        experience,
        asset_end_of_previous_period,
        income_shock_previous_period,
        params,
        model_specs,
    ):
        base = budget_constraint_cont_exp(
            period=period,
            lagged_choice=lagged_choice,
            experience=experience,
            asset_end_of_previous_period=asset_end_of_previous_period,
            income_shock_previous_period=income_shock_previous_period,
            params=params,
            model_specs=model_specs,
        )
        return base + 0.5 * choice

    baseline_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    baseline_solved = baseline_model.solve(params)

    choice_dependent_funcs = dict(model_funcs)
    choice_dependent_funcs["budget_constraint"] = budget_constraint_using_choice
    choice_dependent_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **choice_dependent_funcs
    )
    choice_dependent_solved = choice_dependent_model.solve(params)

    baseline_value = np.asarray(baseline_solved.value)
    choice_dependent_value = np.asarray(choice_dependent_solved.value)
    finite = np.isfinite(baseline_value) & np.isfinite(choice_dependent_value)
    assert not np.allclose(baseline_value[finite], choice_dependent_value[finite])
