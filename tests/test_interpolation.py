"""
This module tests interpolation functions from dcegm.interpolate:
- linear_interpolation_with_extrapolation,
- linear_interpolation_with_inserting_missing_values.
The results are compared to the ones from scipy's linear interpolation function interp1d.
"""


import numpy as np
from scipy.interpolate import interp1d
from dcegm.interpolate import linear_interpolation_with_extrapolation, linear_interpolation_with_inserting_missing_values
from numpy.testing import assert_allclose


"""draw random test arrays from a uniform distribution"""
# fix length of input arrays
n = 5
m = 7
# random test arrays
x = np.random.rand(n)
y = np.random.rand(n)
x_new = np.random.rand(m)
missing_value = np.random.rand(1)


# test linear_interpolation_with_extrapolation
def test_linear_interpolation_with_extrapolation():
    assert_allclose(linear_interpolation_with_extrapolation(x, y, x_new),
                    interp1d(x, y, fill_value="extrapolate")(x_new), atol=1e-10)

# test linear_interpolation_with_missing_values
def test_linear_interpolation_with_missing_values():
    assert_allclose(linear_interpolation_with_inserting_missing_values(x, y, x_new, missing_value),
                interp1d(x, y, bounds_error=False, fill_value=missing_value)(x_new), atol=1e-10)
