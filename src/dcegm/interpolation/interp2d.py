"""Jax implementation of 2D interpolation."""

from typing import Callable, Dict

import jax.lax
import jax.numpy as jnp

from dcegm.interpolation.interp1d import get_index_high_and_low

TRANSFORMATION_MAT = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
    ]
)


def interp2d_policy_and_value_on_wealth_and_regular_grid(
    regular_grid: jnp.ndarray,
    wealth_grid: jnp.ndarray,
    policy_grid: jnp.ndarray,
    value_grid: jnp.ndarray,
    regular_point_to_interp: jnp.ndarray | float,
    wealth_point_to_interp: jnp.ndarray | float,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    params: dict,
    discount_factor,
):
    """Linear 2D interpolation on two grids where wealth has irregular spacing.

    The function interpolates by mapping the irregular quadrilateral defined by the grid
    coordinates into a canonical unit square to simplify the interpolation process.

    Note, while the regular grid has a shape of (n_regular_grid_points,) and only
    contains the values of the regular grid, the irregular wealth grid is different
    for each regular grid point. The same is true for the policy and value grids.
    These three objects together define the 2D value and policy functions.
    Hence, each of them has shape (n_regular_grid_points, n_wealth_grid_points).

    Args:
        regular_grid (jnp.ndarray): A 1d array of shape (n_regular_grid_points,) with
            the values of the regular grid.
        wealth_grid (jnp.ndarray): A 2d array of with the values of the irregular
            wealth grid over the regular grid points of shape
            (n_regular_grid_points, n_wealth_grid_points).
        policy_grid (jnp.ndarray): A 2d array with the policy values of shape
            (n_regular_grid_points, n_wealth_grid_points).
        value_grid (jnp.ndarray): A 2d array with the value function values of shape.
            (n_regular_grid_points, n_wealth_grid_points).
        regular_point_to_interp (jnp.ndarray | float): The regular point for which to
            interpolate the policy and value function.
        wealth_point_to_interp (jnp.ndarray | float): The wealth point for which to
            interpolate the policy and value function.
        compute_utility (Callable): User function to compute the utility of consumption.
        params (dict): A dictionary containing the model parameters.

    Returns:
        tuple: A tuple containing the interpolated values of the policy and
            value function.

    """

    regular_points, wealth_points, coords_idxs = find_grid_coords_for_interp(
        regular_grid=regular_grid,
        wealth_grid=wealth_grid,
        regular_point_to_interp=regular_point_to_interp,
        wealth_point_to_interp=wealth_point_to_interp,
    )

    policy_interp = interp2d_policy(
        regular_points=regular_points,
        wealth_points=wealth_points,
        policy_grid=policy_grid,
        coords_idxs=coords_idxs,
        regular_point_to_interp=regular_point_to_interp,
        wealth_point_to_interp=wealth_point_to_interp,
    )

    value_interp = interp2d_value_and_check_creditconstraint(
        regular_points=regular_points,
        wealth_points=wealth_points,
        value_grid=value_grid,
        coords_idxs=coords_idxs,
        regular_point_to_interp=regular_point_to_interp,
        wealth_point_to_interp=wealth_point_to_interp,
        compute_utility=compute_utility,
        wealth_min_unconstrained=wealth_grid[:, 1],
        value_at_zero_wealth=value_grid[:, 0],
        state_choice_vec=state_choice_vec,
        params=params,
        discount_factor=discount_factor,
    )

    return policy_interp, value_interp


def interp2d_value_on_wealth_and_regular_grid(
    regular_grid: jnp.ndarray,
    wealth_grid: jnp.ndarray,
    value_grid: jnp.ndarray,
    regular_point_to_interp: jnp.ndarray | float,
    wealth_point_to_interp: jnp.ndarray | float,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    params: dict,
    discount_factor,
):
    """Interpolate the value function on a 2D grid.

    This function interpolates the values of the value function at a specific point
    given by `regular_point_to_interp` and `wealth_point_to_interp`.

    The function is useful in situtions where only the value but not the policy needs
    to be computed - such as maximum likelihood estimation.

    Args:
        regular_points (jnp.ndarray): A 1d array of four elements representing the
            regular grid points used for interpolation.
        wealth_points (jnp.ndarray): A 1d array of four elements representing the
            wealth grid points used for interpolation.
        policy_grid (jnp.ndarray): A 2d array of policy function values with shape
            (n_regular_grid_points, n_wealth_grid_points).
        coords_idxs (jnp.ndarray): A 2d array of shape (2, 2) containing the indices
            of the (regular, wealth) grid where the interpolation is performed.
        regular_point_to_interp (float | jnp.ndarray): The regular grid point at which
            to interpolate.
        wealth_point_to_interp (float | jnp.ndarray): The wealth grid point at which
            to interpolate.
        compute_utility (Callable): User-defined function to compute the utility of
            consumption.
        state_choice_vec (Dict[str, int]): Dictionary specifying the state choices.
        params (dict): A dictionary containing model parameters.

    Returns:
        float: The interpolated value of the policy function at the given
            (regular, wealth) point.

    """

    regular_points, wealth_points, coords_idxs = find_grid_coords_for_interp(
        regular_grid=regular_grid,
        wealth_grid=wealth_grid,
        regular_point_to_interp=regular_point_to_interp,
        wealth_point_to_interp=wealth_point_to_interp,
    )
    value_interp = interp2d_value_and_check_creditconstraint(
        regular_points=regular_points,
        wealth_points=wealth_points,
        value_grid=value_grid,
        coords_idxs=coords_idxs,
        regular_point_to_interp=regular_point_to_interp,
        wealth_point_to_interp=wealth_point_to_interp,
        compute_utility=compute_utility,
        wealth_min_unconstrained=wealth_grid[:, 1],
        value_at_zero_wealth=value_grid[:, 0],
        state_choice_vec=state_choice_vec,
        params=params,
        discount_factor=discount_factor,
    )
    return value_interp


def interp2d_policy_on_wealth_and_regular_grid(
    regular_grid: jnp.ndarray,
    wealth_grid: jnp.ndarray,
    policy_grid: jnp.ndarray,
    regular_point_to_interp: jnp.ndarray | float,
    wealth_point_to_interp: jnp.ndarray | float,
):
    """Interpolate the policy function on a 2D grid.

    This function interpolates the values of the policy function at a specific point
    given by `regular_point_to_interp` and `wealth_point_to_interp`.

    The function is useful in situtions where only the policy but not the value needs
    to be computed - such as maximum likelihood estimation.

    Args:
        regular_points (jnp.ndarray): A 1d array of four elements representing the
            regular grid points used for interpolation.
        wealth_points (jnp.ndarray): A 1d array of four elements representing the
            wealth grid points used for interpolation.
        policy_grid (jnp.ndarray): A 2d array of policy function values with shape
            (n_regular_grid_points, n_wealth_grid_points).
        coords_idxs (jnp.ndarray): A 2d array of shape (2, 2) containing the indices
            of the (regular, wealth) grid where the interpolation is performed.
        regular_point_to_interp (float | jnp.ndarray): The regular grid point at which
            to interpolate.
        wealth_point_to_interp (float | jnp.ndarray): The wealth grid point at which
            to interpolate.

    Returns:
        float: The interpolated value of the policy function at the given
            (regular, wealth) point.

    """

    regular_points, wealth_points, coords_idxs = find_grid_coords_for_interp(
        regular_grid=regular_grid,
        wealth_grid=wealth_grid,
        regular_point_to_interp=regular_point_to_interp,
        wealth_point_to_interp=wealth_point_to_interp,
    )

    policy_interp = interp2d_policy(
        regular_points=regular_points,
        wealth_points=wealth_points,
        policy_grid=policy_grid,
        coords_idxs=coords_idxs,
        regular_point_to_interp=regular_point_to_interp,
        wealth_point_to_interp=wealth_point_to_interp,
    )

    return policy_interp


def interp2d_policy(
    regular_points,
    wealth_points,
    policy_grid,
    coords_idxs,
    regular_point_to_interp,
    wealth_point_to_interp,
):
    """Interpolate the policy function on a 2D grid.

    This function interpolates the values of the policy function at a specific point
    given by `regular_point_to_interp` and `wealth_point_to_interp`.

    Args:
        regular_points (jnp.ndarray): A 1d array of four elements representing the
            regular grid points used for interpolation.
        wealth_points (jnp.ndarray): A 1d array of four elements representing the
            wealth grid points used for interpolation.
        policy_grid (jnp.ndarray): A 2d array of policy function values with shape
            (n_regular_grid_points, n_wealth_grid_points).
        coords_idxs (jnp.ndarray): A 2d array of shape (2, 2) containing the indices
            of the (regular, wealth) grid where the interpolation is performed.
        regular_point_to_interp (float | jnp.ndarray): The regular grid point at which
            to interpolate.
        wealth_point_to_interp (float | jnp.ndarray): The wealth grid point at which
            to interpolate.

    Returns:
        float: The interpolated value of the policy function at the given
            (regular, wealth) point.

    """

    policy_known = policy_grid[coords_idxs[:, 0], coords_idxs[:, 1]]

    policy_interp = interp2d(
        x_coords=regular_points,
        y_coords=wealth_points,
        z_vals=policy_known,
        x_new=regular_point_to_interp,
        y_new=wealth_point_to_interp,
    )

    return policy_interp


def interp2d_value_and_check_creditconstraint(
    regular_points,
    wealth_points,
    value_grid,
    coords_idxs,
    regular_point_to_interp,
    wealth_point_to_interp,
    compute_utility,
    wealth_min_unconstrained,
    value_at_zero_wealth,
    state_choice_vec,
    params,
    discount_factor,
):
    """Interpolate the value function on a 2D grid and check for credit constraints.

    This function interpolates the value function at a specific point given by
    `regular_point_to_interp` and `wealth_point_to_interp`. It checks whether the
    point lies within a credit-constrained region and adjusts the interpolation
    accordingly using closed-form solutions for the value of consuming all wealth.

    Args:
        regular_points (jnp.ndarray): A 1d array of four elements representing the
            regular grid points used for interpolation.
        wealth_points (jnp.ndarray): A 1d array of four elements representing the
            wealth grid points used for interpolation.
        policy_grid (jnp.ndarray): A 2d array of policy function values with shape
            (n_regular_grid_points, n_wealth_grid_points).
        coords_idxs (jnp.ndarray): A 2d array of shape (2, 2) containing the indices
            of the (regular, wealth) grid where the interpolation is performed.
        regular_point_to_interp (float | jnp.ndarray): The regular grid point at which
            to interpolate.
        wealth_point_to_interp (float | jnp.ndarray): The wealth grid point at which
            to interpolate.
        compute_utility (callable): A function to compute the utility of consumption.
        wealth_min_unconstrained (jnp.ndarray): A 1d array of minimum unconstrained
            wealth levels for each regular grid point.
        value_at_zero_wealth (jnp.ndarray): A 1d array of value function values at
            zero wealth for each regular grid point.
        params (dict): Dictionary containing the model parameters.

    Returns:
        float: The interpolated value of the value function at the given
            (regular, wealth) point.

    """
    # Check if we are in the credit constrained region
    regular_idx_left = coords_idxs[0, 0]
    credit_constr_left = (
        wealth_point_to_interp <= wealth_min_unconstrained[regular_idx_left]
    )

    # Now recalculate the closed-form value of consuming all wealth
    value_calc_left = (
        compute_utility(
            consumption=wealth_point_to_interp,
            params=params,
            continuous_state=regular_point_to_interp,
            **state_choice_vec,
        )
        + discount_factor * value_at_zero_wealth[regular_idx_left]
    )

    regular_idx_right = coords_idxs[1, 0]
    credit_constr_right = (
        wealth_point_to_interp <= wealth_min_unconstrained[regular_idx_right]
    )
    value_calc_right = (
        compute_utility(
            consumption=wealth_point_to_interp,
            continuous_state=regular_point_to_interp,
            params=params,
            **state_choice_vec,
        )
        + discount_factor * value_at_zero_wealth[regular_idx_right]
    )

    # Select the known values on the grid
    value_known = value_grid[coords_idxs[:, 0], coords_idxs[:, 1]]

    # Overwrite entries if agent is in the credit constrained region
    value_known = jnp.array(
        [
            jax.lax.select(credit_constr_left, value_calc_left, value_known[0]),
            jax.lax.select(credit_constr_right, value_calc_right, value_known[1]),
            jax.lax.select(credit_constr_right, value_calc_right, value_known[2]),
            jax.lax.select(credit_constr_left, value_calc_left, value_known[3]),
        ]
    )

    value_interp = interp2d(
        x_coords=regular_points,
        y_coords=wealth_points,
        z_vals=value_known,
        x_new=regular_point_to_interp,
        y_new=wealth_point_to_interp,
    )

    return value_interp


def find_grid_coords_for_interp(
    regular_grid, wealth_grid, regular_point_to_interp, wealth_point_to_interp
):
    """Find the coordinates and indices of the 2D grid on which we interpolate.

    This function determines the four grid points and their indices for interpolation.
    It exploits the fact that one dimension of the 2D grid is regular. The wealth grid
    is irregular.

    Args:
        regular_grid (jnp.ndarray): A 1d array of shape (n_regular_grid_points,) with
            the values of the regular grid.
        wealth_grid (jnp.ndarray): A 2d array with the values of the irregular wealth
            grid of shape (n_regular_grid_points, n_wealth_grid_points).
        regular_point_to_interp (jnp.ndarray | float): The regular point for which to
            interpolate the policy and value function.
        wealth_point_to_interp (jnp.ndarray | float): The wealth point for which to
            interpolate the policy and value function.


    Returns:
        tuple:

        - regular_points (jnp.ndarray): A 1d array of shape (4,) with the four points
            of the regular grid.
        - wealth_points (jnp.ndarray): A 1d array of shape (4,) with the four points
            of the wealth grid.
        - coords_idxs (jnp.ndarray): A 2d array of shape (2, 2) containing the indices
            of the (regular, wealth) grid where the interpolation is performed.

        The structure of the return arrays are the same in each row (or element):

        - The first element is the lower left point.
        - The second element is the lower right point.
        - The third element is the upper right point.
        - The fourth element is the upper left point.

    """

    # Determine the closest points in the regular direction
    regular_idx_right, regular_idx_left = get_index_high_and_low(
        regular_grid, regular_point_to_interp
    )

    regular_points = jnp.array(
        [
            regular_grid[regular_idx_left],  # lower left
            regular_grid[regular_idx_right],  # lower right
            regular_grid[regular_idx_right],  # upper right
            regular_grid[regular_idx_left],  # upper left
        ]
    )

    # Determine the closest points in the irregular (wealth) direction
    wealth_idx_upper_left, wealth_idx_lower_left = get_index_high_and_low(
        wealth_grid[regular_idx_left], wealth_point_to_interp
    )
    wealth_idx_upper_right, wealth_idx_lower_right = get_index_high_and_low(
        wealth_grid[regular_idx_right], wealth_point_to_interp
    )

    coords_idxs = jnp.array(
        [
            [regular_idx_left, wealth_idx_lower_left],  # lower left
            [regular_idx_right, wealth_idx_lower_right],  # lower right
            [regular_idx_right, wealth_idx_upper_right],  # upper right
            [regular_idx_left, wealth_idx_upper_left],  # upper left
        ]
    )

    wealth_points = wealth_grid[coords_idxs[:, 0], coords_idxs[:, 1]]

    return regular_points, wealth_points, coords_idxs


def interp2d(x_coords, y_coords, z_vals, x_new, y_new):
    """Perform linear 2D interpolation on an irregular quadrilateral.

    This function maps the vertices of an irregular quadrilateral onto a canonical unit
    square and performs bilinear interpolation for a given point within this square.

    Args:
        x_coords (jnp.ndarray): A 1d array of four elements representing the
            x-coordinates of the quadrilateral's vertices.
        y_coords (jnp.ndarray): A 1d array of four elements representing the
            y-coordinates of the quadrilateral's vertices.
        z_vals (jnp.ndarray): A 1d array of four elements representing the values at
            the vertices of the quadrilateral.
        x_new (float): The x-coordinate of the point where interpolation is required.
        y_new (float): The y-coordinate of the point where interpolation is required.

    Returns:
        float: The interpolated value at the point (x_new, y_new).

    """

    # Map the irregular quadrilateral onto a canonical unit square
    x_vec, y_vec = TRANSFORMATION_MAT @ x_coords, TRANSFORMATION_MAT @ y_coords

    # Determine the relative coordinates of the point within the unit square
    x_normalized, y_normalized = determine_coordinates_in_unit_square(
        x_new, y_new, x_vec, y_vec
    )

    # Compute the interpolation weights
    weights = compute_vertex_weights(x_normalized, y_normalized)

    return (weights * z_vals).sum()


def determine_coordinates_in_unit_square(x, y, x_vec, y_vec):
    """Determine the relative coordinates of a point within the unit square.

    This function computes the relative coordinates of a point (x, y) within the
    unit square, using transformed basis vectors `x_vec` and `y_vec`.
    It maps the point of an irregular quadrilateral point into a normalized space
    to ease interpolation.

    Args:
        x (float): The x-coordinate of the point to be mapped.
        y (float): The y-coordinate of the point to be mapped.
        x_vec (jnp.ndarray): A 1d array of four elements representing the transformed
            x-coordinates of the quadrilateral's vertices.
        y_vec (jnp.ndarray): A 1d array of two elements representing the transformed
            y-coordinates of the quadrilateral's vertices.

    Returns:
        tuple: A tuple (x_rel, y_rel) representing the relative coordinates of the
            point within the unit square. `x_rel` is the relative position along
            the x-axis, and `y_rel` is the relative position along the y-axis.

    """

    x_rel = (x - x_vec[0]) / x_vec[1]
    y_rel = (y - y_vec[0] - y_vec[1] * x_rel) / (y_vec[2] + y_vec[3] * x_rel)

    return x_rel, y_rel


def compute_vertex_weights(x, y):
    """Compute interpolation weights for a point within the unit square.

    This function calculates the weights for bilinear interpolation based on the
    relative position of a point (x, y) within a canonical unit square.
    These weights are used to determine the contribution of each vertex of the
    square to the interpolated value.

    Args:
        x (float): The relative x-coordinate of the point within the unit square.
        y (float): The relative y-coordinate of the point within the unit square.

    Returns:
        jnp.ndarray: A 1d array of four weights corresponding to the contributions of
            the lower left, lower right, upper right, and upper left vertices of the
            unit square, respectively.

    """

    return jnp.array([(1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y])
