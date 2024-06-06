from numpy.testing import assert_allclose

from dcegm.numerical_integration import quadrature_hermite


def test_normal_distribution():
    draws, weights = quadrature_hermite(20, 1)
    assert_allclose((draws * weights).sum(), 0, atol=1e-16)
