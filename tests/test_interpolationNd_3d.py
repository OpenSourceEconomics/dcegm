import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import griddata

from dcegm.interpolation.interpNd import (
    interpNd_policy,
    interpNd_value_with_cc,
)
from tests.utils.interpNd_helpers import (
    PARAMS,
    get_compute_utility,
    mask_and_assert_allclose,
)

N_TEST_CASES_ND = 10

jax.config.update("jax_enable_x64", True)


def generate_test_cases_nd(
    R, n_test_cases, seed=1234, n_reg_base=20, nW=30, n_points=20
):
    np.random.seed(seed)
    test_cases = {}
    for test_id in range(n_test_cases):
        test_cases[test_id] = {}

        a, b = np.random.uniform(1, 10), np.random.uniform(1, 10)

        def functional_form(w, rs):
            return a + np.log((w + sum(rs)) * b)

        # Regular grids
        n_reg = [max(3, n_reg_base - i // 2) for i in range(R)]  # Vary sizes slightly
        regular_grids = [np.linspace(1e-8, 100, n) for n in n_reg]

        # Irregular wealth grids
        dims = tuple(n_reg)
        irregular_shape = dims + (nW,)
        irregular_grids = np.empty(irregular_shape)
        for idx in np.ndindex(dims):
            irregular_grids[idx] = np.sort(
                np.exp(np.random.uniform(1, np.log(100), nW))
            )

        # Policy and value grids
        policy_grid = np.empty(irregular_shape)
        value_grid = np.empty(irregular_shape)
        for idx in np.ndindex(dims):
            rs = [regular_grids[k][idx[k]] for k in range(R)]
            w = irregular_grids[idx]
            policy_grid[idx] = functional_form(w, rs)
            value_grid[idx] = policy_grid[idx] * 3.5

        # Test points
        test_w = np.random.uniform(30, 40, n_points)
        test_rs = [np.random.choice(regular_grids[i], n_points) for i in range(R)]
        test_points_reg = np.column_stack(test_rs)  # (n_points, R)

        # For SciPy
        grid_points_list = []
        grid_policy_list = []
        grid_value_list = []
        for idx in np.ndindex(dims):
            for k in range(nW):
                r_vals = [regular_grids[m][idx[m]] for m in range(R)]
                grid_points_list.append(r_vals + [irregular_grids[idx][k]])
                grid_policy_list.append(policy_grid[idx][k])
                grid_value_list.append(value_grid[idx][k])

        grid_points = np.array(grid_points_list)
        grid_policy = np.array(grid_policy_list)
        grid_value = np.array(grid_value_list)

        # Query points
        query_points = np.hstack((test_points_reg, test_w[:, None]))

        compute_utility = get_compute_utility()

        test_cases[test_id].update(
            dict(
                regular_grids=regular_grids,
                irregular_grids=irregular_grids,
                policy_grid=policy_grid,
                value_grid=value_grid,
                test_w=test_w,
                test_points_reg=test_points_reg,
                grid_points=grid_points,
                grid_policy=grid_policy,
                grid_value=grid_value,
                query_points=query_points,
                compute_utility=compute_utility,
            )
        )

    return test_cases


@pytest.fixture(scope="module")
def test_cases_3d():
    return generate_test_cases_nd(R=2, n_test_cases=N_TEST_CASES_ND)


@pytest.mark.parametrize("test_id", range(N_TEST_CASES_ND))
def test_interpNd_policy_against_scipy_3d(test_cases_3d, test_id):
    test_case = test_cases_3d[test_id]
    regular_grids = test_case["regular_grids"]
    wealth_grid = jnp.array(test_case["irregular_grids"])
    policy_grid = jnp.array(test_case["policy_grid"])
    test_w = test_case["test_w"]
    test_points_reg = test_case["test_points_reg"]
    grid_points = test_case["grid_points"]
    grid_policy = test_case["grid_policy"]
    query_points = test_case["query_points"]

    policy_interp_scipy = griddata(
        grid_points, grid_policy, query_points, method="linear"
    )

    interpNd_partial = lambda reg, w: interpNd_policy(
        [jnp.array(g) for g in regular_grids],
        wealth_grid,
        policy_grid,
        jnp.array(reg),
        w,
    )

    policy_interp_nd = jax.vmap(interpNd_partial)(
        test_points_reg,
        jnp.array(test_w),
    )

    mask_and_assert_allclose(
        policy_interp_nd,
        policy_interp_scipy,
        rtol=1e-4,
        atol=1e-4,
        skip_msg="All query points outside convex hull for SciPy (3D policy)",
    )


@pytest.mark.parametrize("test_id", range(N_TEST_CASES_ND))
def test_interpNd_value_against_scipy_3d_no_cc(test_cases_3d, test_id):
    test_case = test_cases_3d[test_id]
    regular_grids = test_case["regular_grids"]
    wealth_grid = jnp.array(test_case["irregular_grids"])
    value_grid = jnp.array(test_case["value_grid"])
    test_w = test_case["test_w"]
    test_points_reg = test_case["test_points_reg"]
    grid_points = test_case["grid_points"]
    grid_value = test_case["grid_value"]
    query_points = test_case["query_points"]
    compute_utility = test_case["compute_utility"]

    # Compute max w_min and mask above to avoid CC in comparison with SciPy
    max_w_min = np.max(test_case["irregular_grids"][..., 1])
    mask = test_w > max_w_min + 1
    if np.sum(mask) == 0:
        pytest.skip("No points above max_w_min")

    value_interp_scipy = griddata(
        grid_points, grid_value, query_points[mask], method="linear"
    )

    state_choice_vec = {"choice": 0}
    discount_factor = PARAMS["discount_factor"]

    interpNd_partial = lambda reg, w: interpNd_value_with_cc(
        [jnp.array(g) for g in regular_grids],
        wealth_grid,
        value_grid,
        jnp.array(reg),
        w,
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=PARAMS,
        discount_factor=discount_factor,
    )

    value_interp_nd = jax.vmap(interpNd_partial)(
        test_points_reg[mask],
        jnp.array(test_w[mask]),
    )

    mask_and_assert_allclose(
        value_interp_nd,
        value_interp_scipy,
        rtol=1e-4,
        atol=1e-4,
        skip_msg="All query points outside convex hull for SciPy (3D value)",
    )
