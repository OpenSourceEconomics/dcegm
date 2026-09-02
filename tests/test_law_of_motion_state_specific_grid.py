"""Tests for the on-demand representative-parent grid selection in law_of_motion.py."""

import jax.numpy as jnp
import numpy as np
from jax import vmap

from dcegm.law_of_motion import (
    _continuous_state_next_period_for_one_state,
    _get_continuous_state_next_period,
    compute_own_continuous_grid_combos,
)


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
        state_space_dict,  # grid_source_state_choice_vec == state_space_dict: no parent/child distinction being tested here, see the dedicated test for that below.
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
        state_space_dict=state_space_dict,
        grid_source_state_choice_vec=state_space_dict,
        additional_continuous_state_names=[],
        params={},
        model_funcs={},
    )
    assert result["dummy_cont"].shape == (2, 1)


def test_get_continuous_state_next_period_uses_grid_source_not_state_space_dict():
    # state_space_dict is the CHILD's own identity (used for the transition function
    # call itself); grid_source_state_choice_vec is a representative PARENT's
    # identity (used only to pick which grid to feed in). These must be allowed to
    # differ -- this is the exact bug this test guards against: using the child's
    # identity to select the grid instead of the parent's.
    state_space_dict = {"group": jnp.array([9, 9])}  # child's own group -- irrelevant
    grid_source_state_choice_vec = {
        "group": jnp.array([0, 1])
    }  # representative parent's group

    def grid_func(group):
        return jnp.where(group == 0, jnp.array([0.0, 1.0]), jnp.array([10.0, 11.0]))

    def compute_continuous_state(group, experience, params):
        return {"experience": experience}

    result = _get_continuous_state_next_period(
        has_additional_continuous_states=True,
        state_space_dict=state_space_dict,
        grid_source_state_choice_vec=grid_source_state_choice_vec,
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
