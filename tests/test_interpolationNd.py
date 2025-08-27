import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal as aaae
from scipy.interpolate import griddata

from dcegm.interpolation.interp2d import (
    interp2d_policy_on_wealth_and_regular_grid,
    interp2d_value_on_wealth_and_regular_grid,
)
from dcegm.interpolation.interpNd import (
    interpNd_policy,
    interpNd_value_with_cc,
)
from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)


# Define utility_crra for testing
def utility_crra(consumption, params, continuous_state=None, **kwargs):
    rho = params["rho"]
    if rho == 1:
        return jnp.log(consumption)
    else:
        return (consumption ** (1 - rho) - 1) / (1 - rho)


# Example params
PARAMS = {
    "discount_factor": 0.95,
    "rho": 0.5,
    "delta": -1,
    "interest_rate": 0.05,
    "consumption_floor": 100,
    "pension": 1000,
    "labor_income": 2000,
}

N_TEST_CASES = 20


@pytest.fixture(scope="module")
def test_cases_2d():
    test_cases = {}
    np.random.seed(1234)

    for test_id in range(N_TEST_CASES):
        test_cases[test_id] = {}

        # setup a functional form
        a, b = np.random.uniform(1, 10), np.random.uniform(1, 10)

        def functional_form(x, y):
            return a + np.log((x + y) * b)

        # create x_grids
        irregular_grids = np.empty((10, 100))
        for k in range(10):
            irregular_grids[k, :] = np.sort(
                np.exp(np.random.uniform(1, np.log(100), 100))
            )
        regular_grid = np.linspace(1e-8, 100, 10)
        regular_grids = np.column_stack([regular_grid for i in range(100)])

        policy = functional_form(irregular_grids, regular_grids)
        value = functional_form(irregular_grids, regular_grids) * 3.5

        test_x = np.random.uniform(30, 40, 44)
        test_y = np.random.choice(regular_grid, 44)
        test_points = np.column_stack((test_x, test_y))

        # transform input values for scipy griddata interpolation routine
        griddata_grids = np.column_stack(
            (irregular_grids.flatten(), regular_grids.flatten())
        )
        griddata_true_values = policy.flatten()

        compute_utility = determine_function_arguments_and_partial_model_specs(
            utility_crra,
            model_specs={},
            continuous_state_name="continuous_state",
        )

        test_cases[test_id]["test_points"] = test_points
        test_cases[test_id]["policy"] = policy
        test_cases[test_id]["value"] = value
        test_cases[test_id]["irregular_grids"] = irregular_grids
        test_cases[test_id]["regular_grid"] = regular_grid
        test_cases[test_id]["griddata_grids"] = griddata_grids
        test_cases[test_id]["griddata_true_values"] = griddata_true_values
        test_cases[test_id]["test_x"] = test_x
        test_cases[test_id]["test_y"] = test_y
        test_cases[test_id]["compute_utility"] = compute_utility

    return test_cases


@pytest.mark.parametrize("test_id", range(N_TEST_CASES))
def test_interpNd_policy_matches_2d_impl(test_cases_2d, test_id):
    test_case = test_cases_2d[test_id]
    policy = test_case["policy"]
    irregular_grids = test_case["irregular_grids"]
    regular_grid = test_case["regular_grid"]
    test_x = test_case["test_x"]
    test_y = test_case["test_y"]

    regular_grids = [jnp.array(regular_grid)]
    wealth_grid = jnp.array(irregular_grids)
    policy_grid = jnp.array(policy)

    def interpNd_partial(x_in, y_in):
        pt_regular = jnp.array([y_in])
        return interpNd_policy(
            regular_grids,
            wealth_grid,
            policy_grid,
            pt_regular,
            x_in,
        )

    policy_interp_nd = jax.vmap(interpNd_partial)(
        jnp.array(test_x),
        jnp.array(test_y),
    )

    interp2d_partial = lambda x_in, y_in: interp2d_policy_on_wealth_and_regular_grid(
        regular_grid=jnp.array(regular_grid),
        wealth_grid=jnp.array(irregular_grids),
        policy_grid=jnp.array(policy),
        wealth_point_to_interp=x_in,
        regular_point_to_interp=y_in,
    )

    policy_interp_2d = jax.vmap(interp2d_partial)(
        jnp.array(test_x),
        jnp.array(test_y),
    )

    assert_allclose(policy_interp_nd, policy_interp_2d, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("test_id", range(N_TEST_CASES))
def test_interpNd_value_matches_2d_impl(test_cases_2d, test_id):
    test_case = test_cases_2d[test_id]
    value = test_case["value"]
    irregular_grids = test_case["irregular_grids"]
    regular_grid = test_case["regular_grid"]
    test_x = test_case["test_x"]
    test_y = test_case["test_y"]
    compute_utility = test_case["compute_utility"]

    regular_grids = [jnp.array(regular_grid)]
    wealth_grid = jnp.array(irregular_grids)
    value_grid = jnp.array(value)

    state_choice_vec = {"choice": 0}
    discount_factor = PARAMS["discount_factor"]

    def interpNd_partial(x_in, y_in):
        pt_regular = jnp.array([y_in])
        return interpNd_value_with_cc(
            regular_grids,
            wealth_grid,
            value_grid,
            pt_regular,
            x_in,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=PARAMS,
            discount_factor=discount_factor,
        )

    value_interp_nd = jax.vmap(interpNd_partial)(
        jnp.array(test_x),
        jnp.array(test_y),
    )

    interp2d_partial = lambda x_in, y_in: interp2d_value_on_wealth_and_regular_grid(
        regular_grid=jnp.array(regular_grid),
        wealth_grid=jnp.array(irregular_grids),
        value_grid=jnp.array(value),
        wealth_point_to_interp=x_in,
        regular_point_to_interp=y_in,
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=PARAMS,
        discount_factor=discount_factor,
    )

    value_interp_2d = jax.vmap(interp2d_partial)(
        jnp.array(test_x),
        jnp.array(test_y),
    )

    assert_allclose(value_interp_nd, value_interp_2d, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("test_id", range(N_TEST_CASES))
def test_interpNd_policy_against_scipy_2d(test_cases_2d, test_id):
    test_case = test_cases_2d[test_id]
    test_points = test_case["test_points"]
    policy = test_case["policy"]
    irregular_grids = test_case["irregular_grids"]
    regular_grid = test_case["regular_grid"]
    griddata_grids = test_case["griddata_grids"]
    griddata_true_values = test_case["griddata_true_values"]
    test_x = test_case["test_x"]
    test_y = test_case["test_y"]

    policy_interp_scipy = griddata(
        griddata_grids, griddata_true_values, test_points, method="linear"
    )

    regular_grids = [jnp.array(regular_grid)]
    wealth_grid = jnp.array(irregular_grids)
    policy_grid = jnp.array(policy)

    def interpNd_partial(x_in, y_in):
        pt_regular = jnp.array([y_in])
        return interpNd_policy(
            regular_grids,
            wealth_grid,
            policy_grid,
            pt_regular,
            x_in,
        )

    policy_interp_nd = jax.vmap(interpNd_partial)(
        jnp.array(test_x),
        jnp.array(test_y),
    )

    assert_allclose(policy_interp_nd, policy_interp_scipy, rtol=1e-5, atol=1e-5)


# Now for 3D and 6D


def generate_test_cases_nd(
    R, n_test_cases, seed=1234, n_reg_base=5, nW=20, n_points=30
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

        compute_utility = determine_function_arguments_and_partial_model_specs(
            utility_crra,
            model_specs={},
            continuous_state_name="continuous_state",
        )

        test_cases[test_id]["regular_grids"] = regular_grids
        test_cases[test_id]["irregular_grids"] = irregular_grids
        test_cases[test_id]["policy_grid"] = policy_grid
        test_cases[test_id]["value_grid"] = value_grid
        test_cases[test_id]["test_w"] = test_w
        test_cases[test_id]["test_points_reg"] = test_points_reg
        test_cases[test_id]["grid_points"] = grid_points
        test_cases[test_id]["grid_policy"] = grid_policy
        test_cases[test_id]["grid_value"] = grid_value
        test_cases[test_id]["query_points"] = query_points
        test_cases[test_id]["compute_utility"] = compute_utility

    return test_cases


N_TEST_CASES_ND = 5  # Fewer for higher D


@pytest.fixture(scope="module")
def test_cases_3d():
    return generate_test_cases_nd(
        R=2, n_test_cases=N_TEST_CASES_ND, n_reg_base=5, nW=20, n_points=30
    )


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

    assert_allclose(
        policy_interp_nd, policy_interp_scipy, rtol=1e-4, atol=1e-4
    )  # Slightly looser tolerance


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

    # Compute max w_min
    max_w_min = np.max(test_case["irregular_grids"][..., 1])
    mask = test_w > max_w_min + 1  # Ensure above all w_min

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

    assert_allclose(value_interp_nd, value_interp_scipy, rtol=1e-4, atol=1e-4)


@pytest.fixture(scope="module")
def test_cases_6d():
    return generate_test_cases_nd(
        R=5, n_test_cases=N_TEST_CASES_ND, seed=4321, n_reg_base=3, nW=10, n_points=20
    )  # Smaller sizes


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

    assert_allclose(
        policy_interp_nd, policy_interp_scipy, rtol=1e-3, atol=1e-3
    )  # Looser for higher D


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

    # Compute max w_min
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

    assert_allclose(value_interp_nd, value_interp_scipy, rtol=1e-3, atol=1e-3)


# Existing tests


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


# Additional test for credit constraint handling
def test_interpNd_value_with_cc():
    # Simple 2D case (R=1)
    regular_grids = [jnp.array([0.0, 1.0])]
    wealth_grid = jnp.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    value_grid = jnp.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    pt_regular = jnp.array([0.5])
    discount_factor = PARAMS["discount_factor"]

    def simple_util(c, params, continuous_state, **kwargs):
        return c**0.5  # Simple sqrt utility

    # Case 1: w > w_min at both corners
    pt_wealth = 3.0
    got = interpNd_value_with_cc(
        regular_grids,
        wealth_grid,
        value_grid,
        pt_regular,
        pt_wealth,
        compute_utility=simple_util,
        state_choice_vec={"choice": 0},
        params=PARAMS,
        discount_factor=discount_factor,
    )
    # Manual: at left (0.0), interp between 3.0:12.0
    # Right [1.5,2.5,3.5], 3.0 between 2.5 and 3.5, t=(3-2.5)/(3.5-2.5)=0.5, 21 +0.5*(22-21)=21.5
    # Then blend t_reg = (0.5-0)/(1-0)=0.5, 12*0.5 + 21.5*0.5 = (6 + 10.75)=16.75
    expected_unconst = 0.5 * 12.0 + 0.5 * 21.5  # 16.75
    assert_allclose(got, expected_unconst, rtol=1e-6)

    # Case 2: w < w_min at both corners
    pt_wealth = 0.5
    got = interpNd_value_with_cc(
        regular_grids,
        wealth_grid,
        value_grid,
        pt_regular,
        pt_wealth,
        compute_utility=simple_util,
        state_choice_vec={"choice": 0},
        params=PARAMS,
        discount_factor=discount_factor,
    )
    # v_cc left: sqrt(0.5) + beta * 10.0
    # right: sqrt(0.5) + beta * 20.0
    # blend 0.5 each: sqrt(0.5) + beta * (0.5*10 + 0.5*20) = sqrt(0.5) + beta*15
    u = simple_util(0.5, PARAMS, pt_regular)
    expected_cc = u + discount_factor * (0.5 * 10.0 + 0.5 * 20.0)
    assert_allclose(got, expected_cc, rtol=1e-6)

    # Case 3: mixed, w < w_min left (1.0), > w_min right (1.5), but 0.5 <1.0 and <1.5
    # both cc. choose w=1.2, left w_min=1.0, 1.2>1.0, right 1.2<1.5
    pt_wealth = 1.2
    got = interpNd_value_with_cc(
        regular_grids,
        wealth_grid,
        value_grid,
        pt_regular,
        pt_wealth,
        compute_utility=simple_util,
        state_choice_vec={"choice": 0},
        params=PARAMS,
        discount_factor=discount_factor,
    )
    # Left: not cc, interp wealth_left [1,2,3], 1.2 between 1 and 2, t=(1.2-1)/(2-1)=0.2, 10 +0.2*(11-10)=10.2
    # Right: cc, since 1.2 <1.5, u(1.2) + beta*20.0
    # Blend 0.5 * 10.2 + 0.5 * (sqrt(1.2) + beta*20)
    u_right = simple_util(1.2, PARAMS, pt_regular)
    v_right_cc = u_right + discount_factor * 20.0
    expected_mixed = 0.5 * 10.2 + 0.5 * v_right_cc
    assert_allclose(got, expected_mixed, rtol=1e-6)
