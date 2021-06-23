"""Interface for solving a consumption-savings model via the Endogenous Grid Method."""
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.special.orthogonal import roots_sh_legendre

from egm_step import call_egm_step, solve_final_period


def solve_egm(
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_func: Callable,
    marginal_utility_func: Callable,
    inv_marginal_utility_func: Callable,
    compute_value_function: Callable,
    compute_next_period_wealth_matrix: Callable,
    compute_next_period_marg_wealth_matrix: Callable,
):
    """Solves a prototypical consumption-savings model using the EGM algorithm.

    EGM stands for Endogenous Grid Method.

    Args:
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_func (callable): The agent's utility function.
        marginal_utility_func (callable): Marginal utility function.
        inv_marginal_utility_func (callable): Inverse of the marginal utility
            function.
        compute_value_function (callable): Function to compute the agent's value
            function, which is an array of length n_grid_wealth.
        compute_next_period_wealth_matrix (callable): Function to compute next
            period wealth matrix which is an array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        compute_next_period_marg_wealth_matrix (callable): Function to compute next
            period wealth matrix which is an array of all possible next period
            marginal wealths with shape (n_quad_stochastic, n_grid_wealth).

        n_periods (int): Number of time periods.
        n_gridpoints (int): Number of exogenous points in the savings grid.
        max_wealth (int or float): Upper bound on wealth.

    Returns:
        policy (np.array): Multi-dimensional array of choice-specific
            consumption policy. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        value (np.array): Multi-dimensional array of choice-specific values of the
            the value function. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
    """
    max_wealth = params.loc[("assets", "max_wealth"), "value"]
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_points = options["quadrature_points_stochastic"]

    # If only one state, i.e. no discrete choices to make,
    # set choice_range to integer; 1 = "working".
    choice_range = [1] if n_choices < 2 else range(n_choices)

    savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    # Gauss-Legendre (shifted) quadrature over the interval [0,1].
    # Standard Gauss-Legendre quadrature (scipy.special.roots_legendre)
    # integrates over [-1, 1].
    quad_points, quad_weights = roots_sh_legendre(n_quad_points)
    quad_points_normal = norm.ppf(quad_points)

    # Empty containers for consumption policy and value function
    policy, value = _create_multi_dim_arrays(options)

    # Determine consumption policy and value function for final period T
    policy, value = solve_final_period(
        policy, value, savings_grid, params, options, utility_func
    )

    # Start backwards induction from second to last period (T - 1)
    for period in range(n_periods - 2, -1, -1):
        for state in choice_range:
            policy, value = call_egm_step(
                period,
                state,
                policy,
                value,
                savings_grid,
                quad_points_normal,
                quad_weights,
                params,
                options,
                utility_func,
                marginal_utility_func,
                inv_marginal_utility_func,
                compute_value_function,
                compute_next_period_wealth_matrix,
                compute_next_period_marg_wealth_matrix,
            )

    return policy, value


def _create_multi_dim_arrays(options: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create arrays for storing the consumption policy and value function.

    Note that we include one additional grid point (n_grid_wealth + 1) to M,
    since we want to set the first positon (j=0) to M_t = 0 for all time periods.

    Args:
        options(Dict[str, int]):

    Returns:
        (Tuple): Tuple containing

            - policy (np.ndarray): Multi-dimensional array for storing
                optimal consumption policy.
                Shape (n_periods, n_choices, 2, n_grid_wealth + 1),
                where in the last dimension, position [0] contains the endogenous
                grid over wealth M, and [1] stores the corresponding levels of
                (choice-specific) optimal consumption c(M, d).
            - value (np.ndarray): Muli-dimensional array for storing
                the value function.
                Shape (n_periods - 1, n_choices, 2, n_grid_wealth + 1),
                where in the last dimension, position [0] contains the endogenous
                grid over wealth M, and [1] stores corresponding level of the
                value function v(M, d).
    """
    n_grid_wealth = options["grid_points_wealth"]
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]

    policy = np.empty((n_periods, n_choices, 2, n_grid_wealth + 1))
    value = np.empty((n_periods, n_choices, 2, n_grid_wealth + 1))

    return policy, value
