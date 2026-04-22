import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from dcegm.interpolation.interpnd_regular import (
    interpnd_policy_and_value_for_child_states_on_regular_grids,
    interpnd_policy_for_child_states_on_regular_grids,
    interpnd_value_for_child_states_on_regular_grids,
)


@pytest.fixture
def interpnd_inputs():
    # Regular grids for two continuous states + shared wealth grid.
    exp_green_grid = np.array([0.0, 0.3, 0.8], dtype=float)
    exp_red_grid = np.array([0.0, 0.5], dtype=float)
    wealth_grid = np.array([0.0, 2.0, 5.0, 9.0], dtype=float)

    n_child_state_choices = 8
    n_cont_combinations = exp_green_grid.size * exp_red_grid.size
    n_wealth_eval = 5
    n_quad = 6

    rng = np.random.default_rng(123)
    green_span = exp_green_grid.max() - exp_green_grid.min()
    red_span = exp_red_grid.max() - exp_red_grid.min()
    wealth_span = wealth_grid.max() - wealth_grid.min()

    # Include both in-grid and out-of-grid points for extrapolation checks.
    continuous_state_child_states = {
        "exp_green": rng.uniform(
            exp_green_grid.min() - 0.3 * green_span,
            exp_green_grid.max() + 0.3 * green_span,
            size=(n_child_state_choices, n_cont_combinations),
        ),
        "exp_red": rng.uniform(
            exp_red_grid.min() - 0.3 * red_span,
            exp_red_grid.max() + 0.3 * red_span,
            size=(n_child_state_choices, n_cont_combinations),
        ),
    }
    wealth_child_states = rng.uniform(
        wealth_grid.min() - 0.3 * wealth_span,
        wealth_grid.max() + 0.3 * wealth_span,
        size=(n_child_state_choices, n_cont_combinations, n_wealth_eval, n_quad),
    )

    return {
        "exp_green_grid": exp_green_grid,
        "exp_red_grid": exp_red_grid,
        "wealth_grid": wealth_grid,
        "n_child_state_choices": n_child_state_choices,
        "n_cont_combinations": n_cont_combinations,
        "continuous_state_child_states": continuous_state_child_states,
        "wealth_child_states": wealth_child_states,
    }


def _run_interpnd(policy_grid_child_states, value_grid_child_states, inputs):
    return interpnd_policy_for_child_states_on_regular_grids(
        additional_continuous_state_grids={
            "exp_green": jnp.asarray(inputs["exp_green_grid"]),
            "exp_red": jnp.asarray(inputs["exp_red_grid"]),
        },
        wealth_grid=jnp.asarray(inputs["wealth_grid"]),
        policy_grid_child_states=jnp.asarray(policy_grid_child_states),
        value_grid_child_states=jnp.asarray(value_grid_child_states),
        continuous_state_child_states={
            "exp_green": jnp.asarray(
                inputs["continuous_state_child_states"]["exp_green"]
            ),
            "exp_red": jnp.asarray(inputs["continuous_state_child_states"]["exp_red"]),
        },
        wealth_child_states=jnp.asarray(inputs["wealth_child_states"]),
        state_choice_child_states={
            "choice": jnp.zeros(inputs["n_child_state_choices"], dtype=jnp.int32)
        },
        compute_utility=_compute_utility,
        params={"u_scale": 2.0},
        discount_factor=0.95,
    )


def _scipy_expected(policy_grid_child_states, inputs):
    exp_green_grid = inputs["exp_green_grid"]
    exp_red_grid = inputs["exp_red_grid"]
    wealth_grid = inputs["wealth_grid"]
    n_child_state_choices = inputs["n_child_state_choices"]
    n_cont_combinations = inputs["n_cont_combinations"]
    continuous_state_child_states = inputs["continuous_state_child_states"]
    wealth_child_states = inputs["wealth_child_states"]

    expected = np.empty_like(wealth_child_states)
    for i in range(n_child_state_choices):
        policy_grid_nd = policy_grid_child_states[i].reshape(
            exp_green_grid.size, exp_red_grid.size, wealth_grid.size
        )
        interp_kwargs = {
            "method": "linear",
            "bounds_error": False,
            "fill_value": None,
        }
        interp = RegularGridInterpolator(
            (exp_green_grid, exp_red_grid, wealth_grid),
            policy_grid_nd,
            **interp_kwargs,
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


def _compute_utility(consumption, params, **kwargs):
    return consumption ** params["u_scale"]


def _run_interpnd_policy_value(
    policy_grid_child_states, value_grid_child_states, inputs
):
    return interpnd_policy_and_value_for_child_states_on_regular_grids(
        additional_continuous_state_grids={
            "exp_green": jnp.asarray(inputs["exp_green_grid"]),
            "exp_red": jnp.asarray(inputs["exp_red_grid"]),
        },
        wealth_grid=jnp.asarray(inputs["wealth_grid"]),
        policy_grid_child_states=jnp.asarray(policy_grid_child_states),
        value_grid_child_states=jnp.asarray(value_grid_child_states),
        continuous_state_child_states={
            "exp_green": jnp.asarray(
                inputs["continuous_state_child_states"]["exp_green"]
            ),
            "exp_red": jnp.asarray(inputs["continuous_state_child_states"]["exp_red"]),
        },
        wealth_child_states=jnp.asarray(inputs["wealth_child_states"]),
        state_choice_child_states={
            "choice": jnp.zeros(inputs["n_child_state_choices"], dtype=jnp.int32)
        },
        compute_utility=_compute_utility,
        params={"u_scale": 2.0},
        discount_factor=0.95,
    )


def _run_interpnd_value_only(value_grid_child_states, inputs):
    return interpnd_value_for_child_states_on_regular_grids(
        additional_continuous_state_grids={
            "exp_green": jnp.asarray(inputs["exp_green_grid"]),
            "exp_red": jnp.asarray(inputs["exp_red_grid"]),
        },
        wealth_grid=jnp.asarray(inputs["wealth_grid"]),
        value_grid_child_states=jnp.asarray(value_grid_child_states),
        continuous_state_child_states={
            "exp_green": jnp.asarray(
                inputs["continuous_state_child_states"]["exp_green"]
            ),
            "exp_red": jnp.asarray(inputs["continuous_state_child_states"]["exp_red"]),
        },
        wealth_child_states=jnp.asarray(inputs["wealth_child_states"]),
        state_choice_child_states={
            "choice": jnp.zeros(inputs["n_child_state_choices"], dtype=jnp.int32)
        },
        compute_utility=_compute_utility,
        params={"u_scale": 2.0},
        discount_factor=0.95,
    )


def _scipy_expected_value_with_consume_all(value_grid_child_states, inputs):
    exp_green_grid = inputs["exp_green_grid"]
    exp_red_grid = inputs["exp_red_grid"]
    wealth_grid = inputs["wealth_grid"]
    n_child_state_choices = inputs["n_child_state_choices"]
    n_cont_combinations = inputs["n_cont_combinations"]
    continuous_state_child_states = inputs["continuous_state_child_states"]
    wealth_child_states = inputs["wealth_child_states"]

    value_interp = np.empty_like(wealth_child_states)
    value_at_zero_interp = np.empty(
        (n_child_state_choices, n_cont_combinations), dtype=float
    )

    for i in range(n_child_state_choices):
        value_grid_nd = value_grid_child_states[i].reshape(
            exp_green_grid.size, exp_red_grid.size, wealth_grid.size
        )
        interp = RegularGridInterpolator(
            (exp_green_grid, exp_red_grid, wealth_grid),
            value_grid_nd,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        interp_v0 = RegularGridInterpolator(
            (exp_green_grid, exp_red_grid),
            value_grid_nd[..., 0],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        for j in range(n_cont_combinations):
            point_reg = np.array(
                [
                    continuous_state_child_states["exp_green"][i, j],
                    continuous_state_child_states["exp_red"][i, j],
                ]
            )
            value_at_zero_interp[i, j] = interp_v0(point_reg).item()

            for w in range(wealth_child_states.shape[2]):
                for q in range(wealth_child_states.shape[3]):
                    point = np.array(
                        [point_reg[0], point_reg[1], wealth_child_states[i, j, w, q]]
                    )
                    value_interp[i, j, w, q] = interp(point).item()

    consume_all_value = (
        wealth_child_states**2 + 0.95 * value_at_zero_interp[:, :, None, None]
    )
    return np.maximum(value_interp, consume_all_value)


def test_interpnd_regular_policy_random_against_scipy(interpnd_inputs):
    inputs = interpnd_inputs
    rng = np.random.default_rng(321)

    policy_grid_child_states = rng.normal(
        loc=0.0,
        scale=2.0,
        size=(
            inputs["n_child_state_choices"],
            inputs["n_cont_combinations"],
            inputs["wealth_grid"].size,
        ),
    )
    # Set value grid to very high values to ensure policy interpolation dominates and no overwrite happens.
    # Interpolation always yields 1e8 and in consume all we have 0.95 * 1e8
    value_grid_child_states = np.full_like(policy_grid_child_states, 1e8)
    out = _run_interpnd(policy_grid_child_states, value_grid_child_states, inputs)
    expected = _scipy_expected(policy_grid_child_states, inputs)
    np.testing.assert_allclose(np.asarray(out), expected)


def test_interpnd_regular_policy_wrapper_matches_policy_from_joint_path(
    interpnd_inputs,
):
    inputs = interpnd_inputs
    rng = np.random.default_rng(111)

    policy_grid_child_states = rng.normal(
        size=(
            inputs["n_child_state_choices"],
            inputs["n_cont_combinations"],
            inputs["wealth_grid"].size,
        )
    )
    value_grid_child_states = rng.normal(
        loc=-20.0,
        scale=1.0,
        size=(
            inputs["n_child_state_choices"],
            inputs["n_cont_combinations"],
            inputs["wealth_grid"].size,
        ),
    )

    policy_from_wrapper = _run_interpnd(
        policy_grid_child_states,
        value_grid_child_states,
        inputs,
    )
    policy_from_joint, _ = _run_interpnd_policy_value(
        policy_grid_child_states,
        value_grid_child_states,
        inputs,
    )

    np.testing.assert_allclose(
        np.asarray(policy_from_wrapper), np.asarray(policy_from_joint)
    )


def test_interpnd_regular_policy_wrapper_overwrite_case(interpnd_inputs):
    inputs = interpnd_inputs
    rng = np.random.default_rng(222)

    policy_grid_child_states = rng.normal(
        size=(
            inputs["n_child_state_choices"],
            inputs["n_cont_combinations"],
            inputs["wealth_grid"].size,
        )
    )
    # Make value interpolation very low so consume-all dominates.
    value_grid_child_states = np.full_like(policy_grid_child_states, -2_000_000.0)

    policy_out = _run_interpnd(
        policy_grid_child_states,
        value_grid_child_states,
        inputs,
    )

    expected_value = _scipy_expected_value_with_consume_all(
        value_grid_child_states, inputs
    )
    expected_value_interp = _scipy_expected(value_grid_child_states, inputs)
    overwrite_mask = expected_value > expected_value_interp

    assert np.any(overwrite_mask)
    np.testing.assert_allclose(
        np.asarray(policy_out)[overwrite_mask],
        np.asarray(inputs["wealth_child_states"])[overwrite_mask],
    )


def test_interpnd_regular_policy_and_value_random(interpnd_inputs):
    inputs = interpnd_inputs
    rng = np.random.default_rng(654)

    policy_grid_child_states = rng.normal(
        size=(
            inputs["n_child_state_choices"],
            inputs["n_cont_combinations"],
            inputs["wealth_grid"].size,
        )
    )
    value_grid_child_states = rng.normal(
        loc=-20.0,
        scale=1.0,
        size=(
            inputs["n_child_state_choices"],
            inputs["n_cont_combinations"],
            inputs["wealth_grid"].size,
        ),
    )

    policy_out, value_out = _run_interpnd_policy_value(
        policy_grid_child_states,
        value_grid_child_states,
        inputs,
    )

    expected_policy_interp = _scipy_expected(policy_grid_child_states, inputs)
    expected_value = _scipy_expected_value_with_consume_all(
        value_grid_child_states, inputs
    )

    # If consume-all dominates, policy must be overwritten to wealth.
    mask_overwrite = expected_value > _scipy_expected(value_grid_child_states, inputs)
    expected_policy = np.where(
        mask_overwrite, inputs["wealth_child_states"], expected_policy_interp
    )

    np.testing.assert_allclose(np.asarray(value_out), expected_value)
    np.testing.assert_allclose(np.asarray(policy_out), expected_policy)


def test_interpnd_regular_value_only_random(interpnd_inputs):
    inputs = interpnd_inputs
    rng = np.random.default_rng(987)

    value_grid_child_states = rng.normal(
        loc=-15.0,
        scale=2.0,
        size=(
            inputs["n_child_state_choices"],
            inputs["n_cont_combinations"],
            inputs["wealth_grid"].size,
        ),
    )

    value_out = _run_interpnd_value_only(value_grid_child_states, inputs)
    expected_value = _scipy_expected_value_with_consume_all(
        value_grid_child_states, inputs
    )
    np.testing.assert_allclose(np.asarray(value_out), expected_value)
