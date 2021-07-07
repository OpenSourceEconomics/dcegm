"""Interface for the DC-EGM algorithm."""
import copy
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.special.orthogonal import roots_sh_legendre

from src.egm_step import do_egm_step
from src.upper_envelope_step import do_upper_envelope_step


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_func: Callable,
    inv_marginal_utility_func: Callable,
    compute_value_function: Callable,
    compute_expected_value: Callable,
    compute_next_period_marginal_utility: Callable,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Solves a discrete-continuous life-cycle model using the DC-EGM algorithm.

    EGM stands for Endogenous Grid Method.

    Args:
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_func (callable): The agent's utility function.
        inv_marginal_utility_func (callable): Inverse of the marginal utility
            function.
        compute_value_function (callable): Function to compute the agent's value
            function, which is an np.ndarray of shape 
            (n_quad_stochastic * n_grid_wealth,).
        compute_expected_value (callable): Function to compute the agent's
            expected value, which is an np.ndarray of shape (n_grid_wealth,).
        compute_next_period_marginal_utilty (callable): Function to compute the
            marginal utility of the next period, which is an np.ndarray of shape
            (n_grid_wealth,).

    Returns:
        (tuple): Tuple containing
        
        - policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            concurrent local optima for consumption. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which will be 
            set to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the arrays contain the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the (consumption) policy 
            function c(M, d), for each time period and each discrete choice.    
        - value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which will be 
            set to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the array contains the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice.  
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

    # Create nested lists for consumption policy and value function.
    # We cannot use multi-dim np.ndarrays here, since the length of
    # the grid is altered by the Upper Envelope step!
    policy, value = _create_multi_dim_lists(options)
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
        # :func:`~dcgm.solve.solve_final_period`.
        for state in choice_range:
            policy, value, expected_value = do_egm_step(
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
                inv_marginal_utility_func,
                compute_value_function,
                compute_expected_value,
                compute_next_period_marginal_utility,
            )

            if state == 1 and n_choices > 1:
                policy_refined, value_refined = do_upper_envelope_step(
                    policy,
                    value,
                    expected_value,
                    period,
                    params,
                    options,
                    utility_func,
                )

                policy[period][state] = policy_refined
                value[period][state] = value_refined

    return policy, value


def set_first_elements_to_zero(
    policy: np.ndarray, value: np.ndarray, options: Dict[str, int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Sets first elements in endogenous wealth grid and consumption policy to zero.

    Args:
        policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            concurrent local optima for consumption. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which will be 
            set to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the arrays contain the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the (consumption) policy 
            function c(M, d), for each time period and each discrete choice. 
        value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which will be 
            set to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the array contains the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice. 

    Returns:
        (tuple): Tuple containing

        - policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. The first element in the 
            endogenous wealth grid and and the first element in the policy function
            are set to zero.
        - value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. The first element in the endogenous
            wealth grid is set to zero.
    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]

    # Add point M_0 = 0 to the endogenous wealth grid in both the
    # policy and value function arrays
    for period in range(n_periods):
        for state in range(n_choices):
            policy[period][state][0, 0] = 0
            value[period][state][0, 0] = 0

            # Add corresponding consumption point c(M=0, d) = 0
            policy[period][state][1, 0] = 0

    return policy, value


def solve_final_period(
    policy: np.ndarray,
    value: np.ndarray,
    savings_grid: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_func: Callable,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Computes solution to final period for consumption policy and value function.

    Args:
        policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            concurrent local optima for consumption. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which we set
            to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the arrays contain the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the (consumption) policy 
            function c(M, d), for each time period and each discrete choice. 
        value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which we set
            to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the array contains the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice. 
        savings_grid (np.ndarray): Array of shape (n_wealth_grid,) denoting the
            exogenous savings grid.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_func (callable): The agent's utility function.

    Returns:
        (tuple): Tuple containing

        - policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies with the solution for the final 
            period included.
        - value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions with the solution for the final period
            included.
    """
    delta = params.loc[("delta", "delta"), "value"]
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else range(n_choices)

    # In last period, nothing is saved for the next period (since there is none),
    # Hence, everything is consumed, c_T(M, d) = M
    for index, state in enumerate(choice_range):
        policy[n_periods - 1][index][0, 1:] = copy.deepcopy(savings_grid)  # M
        policy[n_periods - 1][index][1, 1:] = copy.deepcopy(
            policy[n_periods - 1][index][0, 1:]
        )  # c(M, d)

        value[n_periods - 1][index][0, 2:] = (
            utility_func(policy[n_periods - 1][index][0, 2:], params) - delta * state
        )
        value[n_periods - 1][index][1, 2:] = (
            utility_func(policy[n_periods - 1][index][1, 2:], params) - delta * state
        )

        value[n_periods - 1][index][:, 2] = 0

    return policy, value


def _create_multi_dim_lists(
    options: Dict[str, int]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create nested list for storing the consumption policy and value function.

    Note that we include one additional grid point (n_grid_wealth + 1) to M,
    since we want to set the first positon (j=0) to M_t = 0 for all time
    periods.
    
    Moreover, the lists have variable length, because the Upper Envelope step
    drops suboptimal points from the original grid and adds new ones (kink
    points as well as the corresponding interpolated values of the consumption
    and value functions).

    Args:
        options (dict): Options dictionary.

     Returns:
        (tuple): Tuple containing
        
        - policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            concurrent local optima for consumption. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which will be 
            set to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the arrays contain the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the (consumption) policy 
            function c(M, d), for each time period and each discrete choice. 
        - value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which will be 
            set to zero (that's why we have n_grid_wealth + 1 initial points). 
            Position [0, :] of the array contains the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice. 
    """
    n_grid_wealth = options["grid_points_wealth"]
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]

    policy = [
        [np.empty((2, n_grid_wealth + 1)) for state in range(n_choices)]
        for period in range(n_periods)
    ]

    value = [
        [np.empty((2, n_grid_wealth + 1)) for state in range(n_choices)]
        for period in range(n_periods)
    ]

    return policy, value
