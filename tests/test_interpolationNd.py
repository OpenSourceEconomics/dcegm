import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import griddata

from dcegm.interpolation.interpNd import (
    interpNd_policy,
    interpNd_value_with_cc,
)

# Example params
PARAMS = {"discount_factor": 0.95}


def aaae(a, b, decimal=7):
    np.testing.assert_array_almost_equal(np.array(a), np.array(b), decimal=decimal)


@pytest.mark.parametrize(
    "regular_grids, wealth_grid, policy_grid, value_grid",
    [
        # One simple test case: 2 regular dims, one irregular dim
        (
            [jnp.linspace(0, 1, 4), jnp.linspace(-1, 1, 3)],
            jnp.stack([jnp.linspace(0, 10, 5)] * (4 * 3)).reshape(4, 3, 5),
            jnp.stack([jnp.linspace(0, 1, 5)] * (4 * 3)).reshape(4, 3, 5),
            jnp.stack([jnp.linspace(10, 20, 5)] * (4 * 3)).reshape(4, 3, 5),
        ),
    ],
)
def test_interpNd_policy_matches_scipy(
    regular_grids, wealth_grid, policy_grid, value_grid
):
    # Choose a point inside the domain
    pt_regular = jnp.array([0.3, 0.5])
    pt_wealth = 4.2

    # SciPy baseline: treat each point as (x0, x1, w)
    grid_points = np.array(
        [
            (x0, x1, w)
            for i0, x0 in enumerate(regular_grids[0])
            for i1, x1 in enumerate(regular_grids[1])
            for w in wealth_grid[i0, i1]
        ]
    )
    grid_values = np.array(
        [
            v
            for i0 in range(wealth_grid.shape[0])
            for i1 in range(wealth_grid.shape[1])
            for v in policy_grid[i0, i1]
        ]
    )

    query_point = np.array([[pt_regular[0], pt_regular[1], pt_wealth]])
    expected = griddata(grid_points, grid_values, query_point, method="linear")[0]

    got = interpNd_policy(
        regular_grids, wealth_grid, policy_grid, pt_regular, pt_wealth
    )
    aaae(got, expected, decimal=6)


def test_interpNd_value_runs(
    regular_grids=[jnp.linspace(0, 1, 4), jnp.linspace(-1, 1, 3)],
    wealth_grid=jnp.stack([jnp.linspace(0, 10, 5)] * (4 * 3)).reshape(4, 3, 5),
    value_grid=jnp.stack([jnp.linspace(10, 20, 5)] * (4 * 3)).reshape(4, 3, 5),
):
    pt_regular = jnp.array([0.3, 0.5])
    pt_wealth = 2.0

    def dummy_util(consumption, params, continuous_state, **kwargs):
        return jnp.log(jnp.maximum(consumption, 1e-8))

    got = interpNd_value_with_cc(
        regular_grids,
        wealth_grid,
        value_grid,
        pt_regular,
        pt_wealth,
        compute_utility=dummy_util,
        state_choice_vec={"choice": 0},
        params=PARAMS,
        discount_factor=PARAMS["discount_factor"],
    )
    # Just test that it returns a scalar and finite
    assert jnp.ndim(got) == 0
    assert jnp.isfinite(got)
