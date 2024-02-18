import numpy as np


def linear_interpolation_with_extrapolation(x, y, x_new):
    """Linear interpolation with extrapolation.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        y (np.ndarray): 1d array of shape (n,) containing the y-values
            corresponding to the x-values.
        x_new (np.ndarray or float): 1d array of shape (m,) or float containing
            the new x-values at which to evaluate the interpolation function.

    Returns:
        np.ndarray or float: 1d array of shape (m,) or float containing
            the new y-values corresponding to the new x-values.
            In case x_new contains values outside of the range of x, these
            values are extrapolated.

    """
    # make sure that the function also works for unsorted x-arrays
    # taken from scipy.interpolate.interp1d
    ind = np.argsort(x, kind="mergesort")
    x = x[ind]
    y = np.take(y, ind)

    ind_high = np.searchsorted(x, x_new).clip(max=(x.shape[0] - 1), min=1)
    ind_low = ind_high - 1

    y_high = y[ind_high]
    y_low = y[ind_low]
    x_high = x[ind_high]
    x_low = x[ind_low]

    interpolate_dist = x_new - x_low
    interpolate_slope = (y_high - y_low) / (x_high - x_low)
    interpol_res = (interpolate_slope * interpolate_dist) + y_low

    return interpol_res


def linear_interpolation_with_inserting_missing_values(x, y, x_new, missing_value):
    """Linear interpolation with inserting missing values.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        y (np.ndarray): 1d array of shape (n,) containing the y-values
            corresponding to the x-values.
        x_new (np.ndarray or float): 1d array of shape (m,) or float containing
            the new x-values at which to evaluate the interpolation function.
        missing_value (np.ndarray or float): Flat array of shape (1,) or float
            to set for values of x_new outside of the range of x.

    Returns:
        np.ndarray or float: 1d array of shape (m,) or float containing the
            new y-values corresponding to the new x-values.
            In case x_new contains values outside of the range of x, these
            values are set equal to missing_value.

    """
    interpol_res = linear_interpolation_with_extrapolation(x, y, x_new)
    where_to_miss = (x_new < x.min()) | (x_new > x.max())
    interpol_res[where_to_miss] = missing_value
    return interpol_res
