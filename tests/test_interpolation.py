"""This module tests the interpolation functions from dcegm.interpolate:

- linear_interpolation_with_extrapolation,
- linear_interpolation_with_inserting_missing_values.
The results are compared to the ones from scipy's linear interpolation
function interp1d.

"""
from __future__ import annotations

import numpy as np
import pytest
from dcegm.interpolate import linear_interpolation_with_extrapolation
from dcegm.interpolate import linear_interpolation_with_inserting_missing_values
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d


@pytest.fixture()
def random_test_data():
    """Draw random test arrays from a uniform distribution."""
    n = 5
    m = 7

    x = np.random.rand(n)
    y = np.random.rand(n)
    x_new = np.random.rand(m)
    missing_value = np.random.rand(1)

    return x, y, x_new, missing_value


def test_linear_interpolation_with_extrapolation(random_test_data):
    x, y, x_new, _ = random_test_data

    got = linear_interpolation_with_extrapolation(x, y, x_new)
    expected = interp1d(x, y, fill_value="extrapolate")(x_new)

    assert_allclose(got, expected, atol=1e-10)


def test_linear_interpolation_with_missing_values(random_test_data):
    x, y, x_new, missing_value = random_test_data

    got = linear_interpolation_with_inserting_missing_values(x, y, x_new, missing_value)
    expected = interp1d(x, y, bounds_error=False, fill_value=missing_value)(x_new)

    assert_allclose(got, expected, atol=1e-10)
