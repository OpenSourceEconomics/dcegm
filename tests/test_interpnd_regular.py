import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from dcegm.interpolation.interpnd_regular import (
    interpnd_policy_for_child_states_on_regular_grids,
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


def _run_interpnd(policy_grid_child_states, inputs):
    return interpnd_policy_for_child_states_on_regular_grids(
        additional_continuous_state_grids={
            "exp_green": jnp.asarray(inputs["exp_green_grid"]),
            "exp_red": jnp.asarray(inputs["exp_red_grid"]),
        },
        wealth_grid=jnp.asarray(inputs["wealth_grid"]),
        policy_grid_child_states=jnp.asarray(policy_grid_child_states),
        continuous_state_child_states={
            "exp_green": jnp.asarray(
                inputs["continuous_state_child_states"]["exp_green"]
            ),
            "exp_red": jnp.asarray(inputs["continuous_state_child_states"]["exp_red"]),
        },
        wealth_child_states=jnp.asarray(inputs["wealth_child_states"]),
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

    out = _run_interpnd(policy_grid_child_states, inputs)
    expected = _scipy_expected(policy_grid_child_states, inputs)
    np.testing.assert_allclose(np.asarray(out), expected)
