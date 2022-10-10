import numpy as np
from scipy.special import roots_hermite
from scipy.special import roots_sh_legendre
from scipy.stats import norm


def quadrature_hermite(n_quad_points, sigma):
    """This is the implementation of the hermite quadrature. It is the specific
    quadrature rule for the normal distribution. As it produces different numeric
    results than in the original dcegm paper, we leave it out for now.

    Args:
        n_quad_points (int): Number of quadrature points.
        sigma (float): Standard deviation of normal distribution.

    Returns:

    """
    # This should be the better quadrature. Leave out for now!
    quad_points, quad_weights = roots_hermite(n_quad_points)
    # Rescale draws and weights
    quad_points_scaled = quad_points * np.sqrt(2) * sigma
    quad_weights *= 1 / np.sqrt(np.pi)
    return quad_points_scaled, quad_weights


def quadrature_legendre(n_quad_points, sigma):
    """This is the implementation of the legendre quadrature. It is shifted from it
    designed application to functions on [0, 1].

    Args:
        n_quad_points (int): Number of quadrature points.
        sigma (float): Standard deviation of normal distribution.

    Returns:
    quad_weights (np.ndarray): Weights associated with the stochastic
            quadrature points of shape (n_quad_points,).

    """
    # Gauss-Legendre (shifted) quadrature over the interval [0,1].
    quad_points, quad_weights = roots_sh_legendre(n_quad_points)
    quad_points_normal = norm.ppf(quad_points) * sigma
    return quad_points_normal, quad_weights
