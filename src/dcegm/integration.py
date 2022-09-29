import numpy as np
from scipy.special import roots_hermite


def quadrature(n_quad_points, sigma):
    # This should be the better quadrature. Leave out for now!
    standard_draws, draw_weights_emax = roots_hermite(n_quad_points)
    # Rescale draws and weights
    draws_emax = standard_draws * np.sqrt(2) * sigma
    draw_weights_emax *= 1 / np.sqrt(np.pi)
    return draws_emax, draw_weights_emax
