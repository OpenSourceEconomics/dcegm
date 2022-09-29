from dcegm.integration import quadrature
from numpy.testing import assert_allclose


def test_normal_distribution():
    draws, weights = quadrature(20, 1)
    assert_allclose((draws * weights).sum(), 0, atol=1e-16)
