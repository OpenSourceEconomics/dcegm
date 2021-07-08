from copy import deepcopy
import numpy as np

from scipy.optimize import brenth as root
from scipy import interpolate


eps = 2.2204e-16


def compute_upper_envelope(segments):
    """Own implementation of upper envelope algorithm.
    
    With the refined monotonic endogenous grid M
    T −2 constructed from the upper en
    velope of the interpolated value functions, we obtain a close approximation to the cor-
    rect optimal consumption rule c T −2 (M 1) as we can see in panel (b) of Figure 1
    
    """
    wealth_grid = np.unique(
        np.concatenate([segments[arr][0].tolist() for arr in range(len(segments))])
    )

    values_interp = np.empty((len(segments), len(wealth_grid)))
    for arr in range(len(segments)):
        segment = segments[arr]
        interpolation_func = interpolate.interp1d(
            x=segment[0], y=segment[1], bounds_error=False, fill_value=-np.inf
        )
        values_interp[arr, :] = interpolation_func(wealth_grid)

    max_values_interp = np.tile(values_interp.max(axis=0), (3, 1))  # need this below
    top_segments = values_interp == max_values_interp[0, :]

    grid_points_upper_env = [wealth_grid[0]]
    values_upper_env = [values_interp[0, 0]]
    intersect_points_upper_env = []
    values_intersect_upper_env = []

    # Index of top segment, starting at first (left-most) grid point
    index_first_segment = np.where(top_segments[:, 0] == 1)[0][0]

    move_right = True

    while move_right:
        index_first_segment = np.where(top_segments[:, 0] == 1)[0][0]

        for i in range(1, len(wealth_grid)):
            index_second_segment = np.where(top_segments[:, i] == 1)[0][0]

            if index_second_segment != index_first_segment:
                first_segment = index_first_segment
                second_segment = index_second_segment
                first_grid_point = wealth_grid[i - 1]
                second_grid_point = wealth_grid[i]

                first_interp_func = interpolate.interp1d(
                    segments[first_segment][0],
                    segments[first_segment][1],
                    bounds_error=False,
                )
                values_first_segment = first_interp_func(
                    [first_grid_point, second_grid_point]
                )

                second_interp_func = interpolate.interp1d(
                    segments[second_segment][0],
                    segments[second_segment][1],
                    bounds_error=False,
                )
                values_second_segment = second_interp_func(
                    [first_grid_point, second_grid_point]
                )

                if np.all(
                    np.isfinite(np.stack([values_first_segment, values_second_segment]))
                ) and np.all(np.abs(values_first_segment - values_second_segment) > 0):
                    intersect_point = root(
                        _subtract_values,
                        first_grid_point,
                        second_grid_point,
                        args=(segments[first_segment], segments[second_segment],),
                    )

                    intersect_interp_func = interpolate.interp1d(
                        segments[first_segment][0],
                        segments[first_segment][1],
                        bounds_error=False,
                    )
                    value_intersect = intersect_interp_func(intersect_point)

                    values_all_segments = np.empty((len(segments), 1))
                    for segment in range(len(segments)):
                        all_segments_interp_func = interpolate.interp1d(
                            segments[segment][0],
                            segments[segment][1],
                            bounds_error=False,
                            fill_value=-np.inf,
                        )
                        values_all_segments[segment] = all_segments_interp_func(
                            intersect_point
                        )

                    index_max_value_intersect = np.where(
                        values_all_segments == values_all_segments.max(axis=0)
                    )[0][0]

                    if (index_max_value_intersect == first_segment) | (
                        index_max_value_intersect == second_segment
                    ):
                        # There are no other functions/correspondences above
                        # Include intersection
                        grid_points_upper_env.append(intersect_point)
                        values_upper_env.append(value_intersect)

                        intersect_points_upper_env.append(intersect_point)
                        values_intersect_upper_env.append(value_intersect)

                        if second_segment == index_second_segment:
                            move_right = False
                        else:
                            first_segment = second_segment
                            first_grid_point = intersect_point
                            second_segment = index_second_segment
                            second_grid_point = wealth_grid[i]
                    else:
                        second_segment = index_max_value_intersect
                        second_grid_point = intersect_point

            # Add point if it lies currently on the highest segment
            if (
                any(abs(segments[index_second_segment][0] - wealth_grid[i]) < eps)
                is True
            ):
                grid_points_upper_env.append(wealth_grid[i])
                values_upper_env.append(max_values_interp[0, i])

            index_first_segment = index_second_segment

        points_upper_env = np.empty((2, len(grid_points_upper_env)))
        points_upper_env[0, :] = grid_points_upper_env
        points_upper_env[1, :] = values_upper_env

        points_to_add = np.empty((2, len(intersect_points_upper_env)))
        points_to_add[0] = intersect_points_upper_env
        points_to_add[1] = values_intersect_upper_env

    return points_upper_env, points_to_add


def _subtract_values(grid_point: float, first_segment, second_segment):
    """

    Args:

    Returns:
    """
    first_interp_func = interpolate.interp1d(
        first_segment[0], first_segment[1], bounds_error=False, fill_value="extrapolate"
    )
    second_interp_func = interpolate.interp1d(
        second_segment[0],
        second_segment[1],
        bounds_error=False,
        fill_value="extrapolate",
    )
    first_segment_interp = first_interp_func(grid_point)
    second_segment_interp = second_interp_func(grid_point)

    diff_segments = first_segment_interp - second_segment_interp

    return diff_segments

