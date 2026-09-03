"""Tests for the per-child-grid n-D regular interpolation (Druedahl-Jorgensen path).

Companion to test_interpnd_regular.py, which validates the original shared-grid
contract (still used by direct callers -- see interpolation/interpnd_regular.py's
docstrings for why the two are kept separate). This file validates the "own grids"
variant used by the solve path once continuous grids are state-choice-specific:
each child is interpolated against its *own* grid, not one grid shared across all
children.

"""

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from dcegm.interpolation.interpnd_regular import (
    interpnd_policy_and_value_for_child_states_on_own_regular_grids,
)


def _compute_utility(consumption, params, **kwargs):
    return consumption ** params["u_scale"]


def _scipy_expected_policy(
    policy_grid_child_states,
    exp_green_grids,
    exp_red_grids,
    wealth_grids,
    continuous_state_child_states,
    wealth_child_states,
):
    n_child_state_choices, n_cont_combinations = continuous_state_child_states[
        "exp_green"
    ].shape
    expected = np.empty_like(wealth_child_states)
    for i in range(n_child_state_choices):
        policy_grid_nd = policy_grid_child_states[i].reshape(
            exp_green_grids[i].size, exp_red_grids[i].size, wealth_grids[i].size
        )
        interp = RegularGridInterpolator(
            (exp_green_grids[i], exp_red_grids[i], wealth_grids[i]),
            policy_grid_nd,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        for j in range(n_cont_combinations):
            for w in range(wealth_child_states.shape[2]):
                for q in range(wealth_child_states.shape[3]):
                    point = np.array(
                        [
                            continuous_state_child_states["exp_green"][i, j],
                            continuous_state_child_states["exp_red"][i, j],
                            wealth_child_states[i, j, w, q],
                        ]
                    )
                    expected[i, j, w, q] = interp(point).item()
    return expected


def test_interpnd_own_grids_matches_scipy_per_child_with_different_grids():
    # Two children with *different* grids for both continuous states and for
    # wealth (child 1's grid is a scaled-up version of child 0's) -- this is
    # exactly the scenario that would silently break if the interpolation axis
    # were wrongly shared across children instead of read per child.
    exp_green_grids = np.array(
        [
            [0.0, 0.3, 0.8],
            [0.0, 0.6, 1.6],
        ]
    )
    exp_red_grids = np.array(
        [
            [0.0, 0.5],
            [0.0, 1.0],
        ]
    )
    wealth_grids = np.array(
        [
            [0.0, 2.0, 5.0, 9.0],
            [0.0, 4.0, 10.0, 18.0],
        ],
        dtype=float,
    )

    n_child_state_choices = 2
    n_cont_combinations = exp_green_grids.shape[1] * exp_red_grids.shape[1]
    n_wealth_eval = 4
    n_quad = 3

    rng = np.random.default_rng(42)

    # Query points within each child's own grid range (including a bit outside for
    # extrapolation coverage), NOT the other child's range.
    continuous_state_child_states = {
        "exp_green": np.stack(
            [
                rng.uniform(-0.2, 1.0, size=n_cont_combinations),
                rng.uniform(-0.2, 2.0, size=n_cont_combinations),
            ]
        ),
        "exp_red": np.stack(
            [
                rng.uniform(-0.2, 0.7, size=n_cont_combinations),
                rng.uniform(-0.2, 1.4, size=n_cont_combinations),
            ]
        ),
    }
    wealth_child_states = np.stack(
        [
            rng.uniform(-1.0, 10.0, size=(n_cont_combinations, n_wealth_eval, n_quad)),
            rng.uniform(-2.0, 20.0, size=(n_cont_combinations, n_wealth_eval, n_quad)),
        ]
    )
    policy_grid_child_states = rng.normal(
        size=(n_child_state_choices, n_cont_combinations, wealth_grids.shape[1])
    )
    # High value grid so policy interpolation dominates, no consume-all overwrite.
    value_grid_child_states = np.full_like(policy_grid_child_states, 1e8)

    policy_out, _ = interpnd_policy_and_value_for_child_states_on_own_regular_grids(
        additional_continuous_state_grids_per_child={
            "exp_green": jnp.asarray(exp_green_grids),
            "exp_red": jnp.asarray(exp_red_grids),
        },
        wealth_grid=jnp.asarray(wealth_grids),
        policy_grid_child_states=jnp.asarray(policy_grid_child_states),
        value_grid_child_states=jnp.asarray(value_grid_child_states),
        continuous_state_child_states={
            k: jnp.asarray(v) for k, v in continuous_state_child_states.items()
        },
        wealth_child_states=jnp.asarray(wealth_child_states),
        state_choice_child_states={
            "choice": jnp.zeros(n_child_state_choices, dtype=jnp.int32)
        },
        compute_utility=_compute_utility,
        params={"u_scale": 2.0},
        discount_factor=0.95,
    )

    expected = _scipy_expected_policy(
        policy_grid_child_states,
        exp_green_grids,
        exp_red_grids,
        wealth_grids,
        continuous_state_child_states,
        wealth_child_states,
    )
    np.testing.assert_allclose(np.asarray(policy_out), expected, rtol=1e-6, atol=1e-8)


def test_interpnd_own_grids_matches_shared_grid_path_when_grids_are_identical():
    # Regression check: when every child's own grid happens to be the same
    # (today's global-grid behavior), the per-child-grid path must reproduce
    # exactly what the original shared-grid function produces.
    from dcegm.interpolation.interpnd_regular import (
        interpnd_policy_and_value_for_child_states_on_regular_grids,
    )

    exp_green_grid = np.array([0.0, 0.3, 0.8])
    exp_red_grid = np.array([0.0, 0.5])
    wealth_grid = np.array([0.0, 2.0, 5.0, 9.0], dtype=float)

    n_child_state_choices = 3
    n_cont_combinations = exp_green_grid.size * exp_red_grid.size

    rng = np.random.default_rng(7)
    continuous_state_child_states = {
        "exp_green": rng.uniform(
            -0.2, 1.0, size=(n_child_state_choices, n_cont_combinations)
        ),
        "exp_red": rng.uniform(
            -0.2, 0.7, size=(n_child_state_choices, n_cont_combinations)
        ),
    }
    wealth_child_states = rng.uniform(
        -1.0, 10.0, size=(n_child_state_choices, n_cont_combinations, 4, 3)
    )
    policy_grid_child_states = rng.normal(
        size=(n_child_state_choices, n_cont_combinations, wealth_grid.size)
    )
    value_grid_child_states = rng.normal(
        loc=-20.0, size=(n_child_state_choices, n_cont_combinations, wealth_grid.size)
    )
    state_choice_child_states = {
        "choice": jnp.zeros(n_child_state_choices, dtype=jnp.int32)
    }

    shared_policy, shared_value = (
        interpnd_policy_and_value_for_child_states_on_regular_grids(
            additional_continuous_state_grids={
                "exp_green": jnp.asarray(exp_green_grid),
                "exp_red": jnp.asarray(exp_red_grid),
            },
            wealth_grid=jnp.asarray(wealth_grid),
            policy_grid_child_states=jnp.asarray(policy_grid_child_states),
            value_grid_child_states=jnp.asarray(value_grid_child_states),
            continuous_state_child_states={
                k: jnp.asarray(v) for k, v in continuous_state_child_states.items()
            },
            wealth_child_states=jnp.asarray(wealth_child_states),
            state_choice_child_states=state_choice_child_states,
            compute_utility=_compute_utility,
            params={"u_scale": 2.0},
            discount_factor=0.95,
        )
    )

    own_grids_policy, own_grids_value = (
        interpnd_policy_and_value_for_child_states_on_own_regular_grids(
            additional_continuous_state_grids_per_child={
                "exp_green": jnp.tile(exp_green_grid, (n_child_state_choices, 1)),
                "exp_red": jnp.tile(exp_red_grid, (n_child_state_choices, 1)),
            },
            wealth_grid=jnp.tile(wealth_grid, (n_child_state_choices, 1)),
            policy_grid_child_states=jnp.asarray(policy_grid_child_states),
            value_grid_child_states=jnp.asarray(value_grid_child_states),
            continuous_state_child_states={
                k: jnp.asarray(v) for k, v in continuous_state_child_states.items()
            },
            wealth_child_states=jnp.asarray(wealth_child_states),
            state_choice_child_states=state_choice_child_states,
            compute_utility=_compute_utility,
            params={"u_scale": 2.0},
            discount_factor=0.95,
        )
    )

    np.testing.assert_array_equal(
        np.asarray(shared_policy), np.asarray(own_grids_policy)
    )
    np.testing.assert_array_equal(np.asarray(shared_value), np.asarray(own_grids_value))
