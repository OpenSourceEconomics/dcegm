"""This module contains an interpolation routine for conducting linear interpolation on
grids where one dimension of the grid mesh is not equidistant."""

import numpy as np

# define auxiliary global
A = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
    ]
).T


def custom_interp2d_quad(x_grids, y_grid, values, points):
    """This function is able to interpolate linearly on a two dimensional grid where one
    dimension has irregular spacing between the grid points.

    Parameters
    ----------
    x_grids: np.array
        Array of x grids for every y grid node.
    y_grid: np.array
        y grid array
    values: np.array
        Array of dimension y_grid length time
    points: np.array
        coordinates at which the interpolation should be conducted
    Returns
    ------
    results: np.array
        Output array that contains the interpolated values

    """

    n_y_grid = y_grid.shape[0]
    n_points = points.shape[0]
    z = np.empty((n_points, 4))
    x_cords = np.empty((n_points, 4))
    y_cords = np.empty((n_points, 4))

    for i in range(n_points):

        x, y = points[i]

        # first determine the index of the regular dimension
        y_indx = search(n_y_grid, y_grid, y)
        # for each grid point of the regular dimension determine the closest
        # point on the irregular direction
        x_indx1 = search(len(x_grids[y_indx]), x_grids[y_indx], x)
        x_indx3 = search(len(x_grids[y_indx + 1]), x_grids[y_indx + 1], x)
        x_cords[i] = [
            x_grids[y_indx][x_indx1],
            x_grids[y_indx + 1][x_indx3],
            x_grids[y_indx + 1][x_indx3 + 1],
            x_grids[y_indx][x_indx1 + 1],
        ]

        y_cords[i, np.array([0, 3])] = y_grid[y_indx]
        y_cords[i, np.array([1, 2])] = y_grid[y_indx + 1]

        z[i] = [
            values[y_indx][x_indx1],
            values[y_indx + 1][x_indx3],
            values[y_indx + 1][x_indx3 + 1],
            values[y_indx][x_indx1 + 1],
        ]

    alpha, beta = (x_cords @ A), (y_cords @ A)

    m, l = calculate_map_params(points[:, 0], points[:, 1], alpha, beta)
    weights = compute_weights(l, m)

    return (weights * z).sum(axis=1)


def custom_interp2d_quad_value_function(
    x_grids, y_grid, values, points, *, float_util, params
):
    """This function is able to interpolate linearly on a two dimensional grid where one
    dimension has irregular spacing between the grid points.

    Parameters
    ----------
    x_grids: np.array
        Array of x grids for every y grid node.
    y_grid: np.array
        y grid array
    values: np.array
        Array of dimension y_grid length time
    points: np.array
        coordinates at which the interpolation should be conducted
    Returns
    ------
    results: np.array
        Output array that contains the interpolated values

    """

    n_y_grid = y_grid.shape[0]
    n_points = points.shape[0]

    z = np.empty((n_points, 4))
    x_cords = np.empty((n_points, 4))
    y_cords = np.empty((n_points, 4))

    for i in range(n_points):

        x, y = points[i]

        # first determine the index of the regular dimension
        y_indx = search(n_y_grid, y_grid, y)
        # for each grid point of the regular dimension determine the closest
        # point on the irregular direction
        x_indx1 = search(len(x_grids[y_indx]), x_grids[y_indx], x)
        x_indx3 = search(len(x_grids[y_indx + 1]), x_grids[y_indx + 1], x)
        x_cords[i] = [
            x_grids[y_indx][x_indx1],
            x_grids[y_indx + 1][x_indx3],
            x_grids[y_indx + 1][x_indx3 + 1],
            x_grids[y_indx][x_indx1 + 1],
        ]

        y_cords[i, np.array([0, 3])] = y_grid[y_indx]
        y_cords[i, np.array([1, 2])] = y_grid[y_indx + 1]

        z[i] = [
            values[y_indx][x_indx1],
            values[y_indx + 1][x_indx3],
            values[y_indx + 1][x_indx3 + 1],
            values[y_indx][x_indx1 + 1],
        ]

        # value in credit-constrained region

        if x < x_grids[y_indx][1]:
            x_cords[i, 0] = x
            # if x == 0:
            #     z[i, 0] = transform(values[y_indx][0], theta)

            # else:
            z[i, 0] = float_util(x, params) + params["beta"] * values[y_indx][0]

        if x < x_grids[y_indx + 1][1]:
            x_cords[i, 1] = x
            # if x == 0:
            #     z[i, 1] = transform(values[y_indx + 1][0], params)

            # else:
            z[i, 1] = float_util(x, params) + params["beta"] * values[y_indx + 1][0]

    alpha, beta = (x_cords @ A), (y_cords @ A)
    m, l = calculate_map_params(points[:, 0], points[:, 1], alpha, beta)
    weights = compute_weights(l, m)

    # return de_transform((weights * z).sum(axis=1), params)
    return (weights * z).sum(axis=1)


def calculate_map_params(x, y, alpha, beta):
    l = (y - beta[:, 0]) / beta[:, 1]
    m = (x - alpha[:, 0] - alpha[:, 1] * l) / (alpha[:, 2] + alpha[:, 3] * l)
    return m, l


def compute_weights(l, m):
    return np.vstack(((1 - l) * (1 - m), l * (1 - m), l * m, (1 - l) * m)).T


def search(grid_length, sorted_grid, element):
    """This function searches the element on the specific grid that is used for the
    interpolation."""

    # if the element is outside the bounds
    # use the border elements for interpolation

    if element <= sorted_grid[0]:
        return 0
    elif element >= sorted_grid[grid_length - 2]:
        return grid_length - 2

    # if within bounds apply binary search algorithm

    # Divide grid length by two and round down
    aux = grid_length // 2
    min_indx = 0
    # while aux != 0 continue searching
    while aux:
        # create a new candidate
        candidate = min_indx + aux

        # if candidate element is smaller or equal to element update min_indx
        if sorted_grid[candidate] <= element:
            min_indx = candidate

        # update grid_length (since we can reject half of the grid candidates)
        grid_length -= aux
        aux = grid_length // 2

    return min_indx
