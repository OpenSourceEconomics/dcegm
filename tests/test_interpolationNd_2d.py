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
from tests.utils.interpNd_helpers import (
    PARAMS,
    get_compute_utility,
    mask_and_assert_allclose,
)

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
        regular_grids = np.column_stack([regular_grid for _ in range(100)])

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

        compute_utility = get_compute_utility()

        test_cases[test_id].update(
            dict(
                test_points=test_points,
                policy=policy,
                value=value,
                irregular_grids=irregular_grids,
                regular_grid=regular_grid,
                griddata_grids=griddata_grids,
                griddata_true_values=griddata_true_values,
                test_x=test_x,
                test_y=test_y,
                compute_utility=compute_utility,
            )
        )

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
    # Mask out points outside convex hull where SciPy returns NaN
    mask_and_assert_allclose(
        policy_interp_nd,
        policy_interp_scipy,
        rtol=1e-5,
        atol=1e-5,
        skip_msg="All query points outside convex hull for SciPy (2D policy)",
    )


# Old tests migrated from the combined ND test file
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


def test_interpNd_value_with_cc():
    # Simple 2D case (R=1)
    regular_grids = [jnp.array([0.0, 1.0])]
    wealth_grid = jnp.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    value_grid = jnp.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    pt_regular = jnp.array([0.5])
    discount_factor = PARAMS["discount_factor"]

    def simple_util(consumption, params, continuous_state, **kwargs):
        return consumption**0.5  # Simple sqrt utility

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
    u = simple_util(consumption=0.5, params=PARAMS, continuous_state=pt_regular)
    expected_cc = u + discount_factor * (0.5 * 10.0 + 0.5 * 20.0)
    assert_allclose(got, expected_cc, rtol=1e-6)

    # Case 3: mixed. The implementation treats w_min as wealth_grid[..., 1].
    # With wealth_grid rows [1.0, 2.0, 3.0] (left) and [1.5, 2.5, 3.5] (right),
    # w_min_left = 2.0 and w_min_right = 2.5. Choose pt_wealth = 2.2 so that
    # left is unconstrained (2.2 > 2.0) and right is constrained (2.2 <= 2.5).
    pt_wealth = 2.2
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
    # Left: not cc, interp wealth_left [1,2,3], 2.2 between 2 and 3,
    #   t=(2.2-2)/(3-2)=0.2, 11 + 0.2*(12-11)=11.2
    # Right: cc, since 2.2 <= 2.5, u(2.2) + beta*20.0
    # Blend: 0.5 * 11.2 + 0.5 * (sqrt(2.2) + beta*20)
    u_right = simple_util(consumption=2.2, params=PARAMS, continuous_state=pt_regular)
    v_right_cc = u_right + discount_factor * 20.0
    expected_mixed = 0.5 * 11.2 + 0.5 * v_right_cc
    assert_allclose(got, expected_mixed, rtol=1e-6)


def test_interpNd_value_with_cc_minimal():
    # Minimal, hand-checkable case (R=1)
    regular_grids = [jnp.array([0.0, 1.0])]
    # w_min_left=1.0, w_min_right=2.0
    wealth_grid = jnp.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    # Linear in wealth: value = 10 * wealth on each row
    value_grid = jnp.array([[0.0, 10.0, 20.0], [0.0, 20.0, 40.0]])
    pt_regular = jnp.array([0.5])
    beta = PARAMS["discount_factor"]

    def simple_util(consumption, params, continuous_state, **kwargs):
        return jnp.sqrt(consumption)

    # Both constrained (w <= 1.0 and <= 2.0)
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
        discount_factor=beta,
    )
    # v_cc_left = sqrt(0.5) + beta * 0.0; v_cc_right = sqrt(0.5) + beta * 0.0
    expected = jnp.sqrt(0.5)
    assert_allclose(got, expected, rtol=1e-8, atol=1e-8)

    # Mixed: left unconstrained (w > 1.0), right constrained (w <= 2.0)
    pt_wealth = 1.5
    got = interpNd_value_with_cc(
        regular_grids,
        wealth_grid,
        value_grid,
        pt_regular,
        pt_wealth,
        compute_utility=simple_util,
        state_choice_vec={"choice": 0},
        params=PARAMS,
        discount_factor=beta,
    )
    # Left (unconstrained): interpolate on [0,1,2] -> value = 10*1.5 = 15.0
    # Right (constrained): v_cc = sqrt(1.5) + beta * 0.0
    # Blend 50/50 across regular axis
    expected = 0.5 * 15.0 + 0.5 * jnp.sqrt(1.5)
    assert_allclose(got, expected, rtol=1e-8, atol=1e-8)
