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

N_TEST_CASES_ND = 1


def generate_test_cases_nd(
    R, n_test_cases, seed=4321, n_reg_base=12, nW=10, n_points=10
):
    np.random.seed(seed)
    test_cases = {}
    for test_id in range(n_test_cases):
        test_cases[test_id] = {}

        a, b = np.random.uniform(1, 10), np.random.uniform(1, 10)

        def functional_form(w, rs):
            return a + np.log((w + sum(rs)) * b)

        # Regular grids
        n_reg = [max(3, n_reg_base - i // 2) for i in range(R)]
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

        # Test points: choose wealth within a safe unconstrained range if possible
        w_min_global_max = np.max(irregular_grids[..., 1])
        w_max_global_min = np.min(irregular_grids[..., -1])
        low = w_min_global_max + 1.0
        high = w_max_global_min - 1.0
        if low < high:
            test_w = np.random.uniform(low, high, n_points)
        else:
            # Fallback to a generic range if degenerate
            test_w = np.random.uniform(30, 40, n_points)
        test_rs = [np.random.choice(regular_grids[i], n_points) for i in range(R)]
        test_points_reg = np.column_stack(test_rs)

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
def test_cases_6d():
    return generate_test_cases_nd(
        R=5, n_test_cases=N_TEST_CASES_ND, n_reg_base=15, nW=20, n_points=5
    )


# Fast smoke tests: run first to ensure outputs are produced with correct shapes
@pytest.mark.parametrize("test_id", range(N_TEST_CASES_ND))
def test_00_interpNd_policy_smoke_6d(test_cases_6d, test_id):
    test_case = test_cases_6d[test_id]
    regular_grids = test_case["regular_grids"]
    wealth_grid = jnp.array(test_case["irregular_grids"])
    policy_grid = jnp.array(test_case["policy_grid"])
    test_w = test_case["test_w"]
    test_points_reg = test_case["test_points_reg"]

    interpNd_partial = lambda reg, w: interpNd_policy(
        [jnp.array(g) for g in regular_grids],
        wealth_grid,
        policy_grid,
        jnp.array(reg),
        w,
    )

    k = min(2, len(test_w))
    policy_interp_nd = jax.vmap(interpNd_partial)(
        test_points_reg[:k],
        jnp.array(test_w[:k]),
    )
    assert policy_interp_nd.shape == (k,)
    assert jnp.all(jnp.isfinite(policy_interp_nd))


@pytest.mark.parametrize("test_id", range(N_TEST_CASES_ND))
def test_01_interpNd_value_smoke_6d_no_cc(test_cases_6d, test_id):
    test_case = test_cases_6d[test_id]
    regular_grids = test_case["regular_grids"]
    wealth_grid = jnp.array(test_case["irregular_grids"])
    value_grid = jnp.array(test_case["value_grid"])
    test_w = test_case["test_w"]
    test_points_reg = test_case["test_points_reg"]
    compute_utility = test_case["compute_utility"]

    # pick a couple of points that are clearly above all w_min to avoid cc branch
    max_w_min = np.max(test_case["irregular_grids"][..., 1])
    mask = test_w > max_w_min + 1
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        pytest.skip("No points above max_w_min")
    take = min(2, idxs.size)
    idxs = idxs[:take]

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
        test_points_reg[idxs],
        jnp.array(test_w[idxs]),
    )
    assert value_interp_nd.shape == (take,)
    assert jnp.all(jnp.isfinite(value_interp_nd))


@pytest.mark.parametrize("test_id", range(N_TEST_CASES_ND))
def test_interpNd_policy_against_scipy_6d(test_cases_6d, test_id):
    test_case = test_cases_6d[test_id]
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
        rtol=1e-3,
        atol=1e-3,
        skip_msg="All query points outside convex hull for SciPy (6D policy)",
    )


@pytest.mark.parametrize("test_id", range(N_TEST_CASES_ND))
def test_interpNd_value_against_scipy_6d_no_cc(test_cases_6d, test_id):
    test_case = test_cases_6d[test_id]
    regular_grids = test_case["regular_grids"]
    wealth_grid = jnp.array(test_case["irregular_grids"])
    value_grid = jnp.array(test_case["value_grid"])
    test_w = test_case["test_w"]
    test_points_reg = test_case["test_points_reg"]
    grid_points = test_case["grid_points"]
    grid_value = test_case["grid_value"]
    query_points = test_case["query_points"]
    compute_utility = test_case["compute_utility"]

    # Build a per-point mask that enforces no-cc at the actual corners used
    irregular_grids = test_case["irregular_grids"]
    w_min_grid = irregular_grids[..., 1]
    dims = tuple(len(g) for g in regular_grids)

    def bracket_indices(grid, x):
        hi = np.searchsorted(grid, x, side="right")
        hi = np.clip(hi, 1, len(grid) - 1)
        lo = hi - 1
        return lo, hi

    # Enumerate corners for R dims as bits
    R = len(regular_grids)
    C = 1 << R
    selectors = np.array(
        [[(k >> r) & 1 for r in range(R)] for k in range(C)], dtype=int
    )

    per_point_mask = []
    for i in range(len(test_w)):
        rs = test_points_reg[i]
        lo_idxs, hi_idxs = zip(
            *(bracket_indices(regular_grids[r], rs[r]) for r in range(R))
        )
        lo_idxs = np.array(lo_idxs)
        hi_idxs = np.array(hi_idxs)
        corner_idx = np.where(selectors == 0, lo_idxs, hi_idxs)  # (C,R)
        # Gather w_min at these corners
        w_min_at_corners = w_min_grid[tuple(corner_idx.T)]  # (C,)
        per_point_mask.append(test_w[i] > np.max(w_min_at_corners) + 1e-12)
    per_point_mask = np.array(per_point_mask, dtype=bool)
    if not np.any(per_point_mask):
        pytest.skip("No points above per-point w_min across used corners")

    value_interp_scipy = griddata(
        grid_points, grid_value, query_points[per_point_mask], method="linear"
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
        test_points_reg[per_point_mask],
        jnp.array(test_w[per_point_mask]),
    )

    mask_and_assert_allclose(
        value_interp_nd,
        value_interp_scipy,
        rtol=1e-3,
        atol=1e-3,
        skip_msg="All query points outside convex hull for SciPy (6D value)",
    )
