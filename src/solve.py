"""Interface for solving a consumption-savings model via the Endogenous Grid Method."""
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.special.orthogonal import roots_sh_legendre

from src.egm_step import call_egm_step, set_first_elements_to_zero, solve_final_period


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
            function, which is an array of shape (n_grid_wealth,).
        compute_next_period_wealth_matrix (callable): Function to compute next
            period wealth matrix which is an array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        compute_next_period_marg_wealth_matrix (callable): Function to compute next
            period wealth matrix which is an array of all possible next period
            marginal wealths with shape (n_quad_stochastic, n_grid_wealth).

    Returns:
        policy (np.ndarray): Multi-dimensional array of choice-specific
            consumption policy. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        value (np.ndarray): Multi-dimensional array of choice-specific values of the
            the value function. Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
    """
    max_wealth = params.loc[("assets", "max_wealth"), "value"]
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_points = options["quadrature_points_stochastic"]

    # If only one state, i.e. no discrete choices to make,
    # set choice_range to 1 = "working".
    choice_range = [1] if n_choices < 2 else range(n_choices)

    savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    # Gauss-Legendre (shifted) quadrature over the interval [0,1].
    # Standard Gauss-Legendre quadrature (scipy.special.roots_legendre)
    # integrates over [-1, 1].
    quad_points, quad_weights = roots_sh_legendre(n_quad_points)
    quad_points_normal = norm.ppf(quad_points)

    # Empty containers for consumption policy and value function
    policy, value = _create_multi_dim_arrays(options)
    policy, value = set_first_elements_to_zero(policy, value, options)
    policy, value = solve_final_period(
        policy, value, savings_grid, params, options, utility_func
    )

    # Start backwards induction from second to last period (T - 1)
    for period in range(n_periods - 2, -1, -1):

        # Note: In the DC-EGM retirement model with two states
        # (0 = "retirement" and 1 = "working") ``state`` denotes both the
        # STATE INDICATOR and the INDEX of the ``policy`` and ``value`` arrays.
        # In fact, in the retirement model, ``states`` dual roles coincide.
        # Meaning, for the "retirement" state, denotes both the state indicator
        # and the index. ``state``'s role as an indicator becomes apparent when
        # subtracting the disutility of work (denoted by ``delta``) from the
        # agent's utility function via `` - state * delta``, which is unequal to 0
        # when then agent is working,
        # see :func:`~dcgm.egm_step.get_current_period_value`.
        # In the EGM consumption-savings model, however, there is no discrete
        # choice to make so that ``state`` is - always - set to 1 ("working").
        # Consequently, ``state``'s roles as indicator and index do not overlap
        # anymore, i.e. the state indicator is 1, but the corresponding index
        # in the ``policy`` and ``value`` arrays is 0!
        # To account for this discrepancy, several one-liners have been put in place
        # to set ``state`` = 0 when it needs to take on the role as an index;
        # see :func:`~dcgm.consumption_savings_model.compute_value_function` and
        # :func:`~dcgm.egm_step.call_egm_step`.
        # Similarly, when we loop over choices, in the consumption-savings model
        # with no discrete choice here, one-liners are in place to keep the
        # existing for-loop structure. In Practice, ``choice_range`` = [1] in
        # this case, see :func:`~dcgm.call_egm_step.get_next_period_value` and
        # :func:`~dcgm.call_egm_step.solve_final_period`.
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
        options (dict): Options dictionary.

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
