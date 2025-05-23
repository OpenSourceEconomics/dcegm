from typing import Tuple

import numpy as np
from scipy.special import roots_hermite, roots_sh_legendre
from scipy.stats import norm


def quadrature_hermite(
    n_quad_points: int, income_shock_std: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Hermite quadrature points and weights.

    It is the specific quadrature rule for the normal distribution.
    As it produces different numeric results than in the original dcegm paper,
    we leave it out for now.

    Args:
        n_quad_points (int): Number of quadrature points.
        income_shock_std (float): Standard deviation of the normal distribution.

    Returns:
        tuple:

        - quad_points_scaled (np.ndarray): 1d array of shape (n_quad_points,)
            containing the Hermite quadrature points.
        - quad_weights (np.ndarray): 1d array of shape (n_quad_points,)
            containing the associated Hermite quadrature weights.

    """
    # This should be the better quadrature. Leave out for now!
    quad_points, quad_weights = roots_hermite(n_quad_points)
    # Rescale draws and weights
    quad_points_scaled = quad_points * np.sqrt(2) * income_shock_std
    quad_weights *= 1 / np.sqrt(np.pi)

    return quad_points_scaled, quad_weights


def quadrature_legendre(n_quad_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Gauss-Legendre quadrature points and weights.

    The stochastic Gauss-Legendre quadrature points are shifted points
    drawn from the [0, 1] interval.

    Args:
        n_quad_points (int): Number of quadrature points.
        income_shock_std (float): Standard deviation of the normal distribution.

    Returns:
        tuple:

        - quad_points_normal (np.ndarray): 1d array of shape (n_quad_points,)
            containing the Hermite quadrature points.
        - quad_weights (np.ndarray): 1d array of shape (n_quad_points,)
            containing the associated stochastic quadrature weights.

    """
    quad_points, quad_weights = roots_sh_legendre(n_quad_points)
    quad_points_normal = norm.ppf(quad_points)

    return quad_points_normal, quad_weights
