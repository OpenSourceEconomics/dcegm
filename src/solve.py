"""Interface for the DC-EGM algorithm."""
import copy
from typing import Callable, Dict, List, Tuple
from functools import partial

import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.stats import norm
from scipy.special.orthogonal import roots_sh_legendre

from src.egm_step import do_egm_step
from src.upper_envelope_step import do_upper_envelope_step


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_functions: Dict[str, callable],
    compute_expected_value: Callable,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Solves a discrete-continuous life-cycle model using the DC-EGM algorithm.
    EGM stands for Endogenous Grid Method.
    Args:
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of (i) utility, (ii) inverse marginal utility, 
            and (iii) next period marginal utility.
        value_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of the agent's (i) value function before
            the final period, (ii) value function in the final period, 
            and (iii) the expected value.
            
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
    exogenous_grid = {
        "savings": savings_grid,
        "quadrature_points": quad_points_normal,
        "quadrature_weights": quad_weights,
    }

    # Create nested lists for consumption policy and value function.
    # We cannot use multi-dim np.ndarrays here, since the length of
    # the grid is altered by the Upper Envelope step!
    policy_arr, value_arr = _create_multi_dim_arrays(options)
    policy_arr, value_arr = solve_final_period(
        policy_arr,
        value_arr,
        savings_grid=savings_grid,
        params=params,
        options=options,
        compute_utility=utility_functions["utility"],
    )

    # Make new function or move inside func:`solve_final_period`
    current_policy_function = dict()
    current_value_function = dict()
    for index, state in enumerate(choice_range):
        final_policy = policy_arr[n_periods - 1, index, :][
            :, ~np.isnan(policy_arr[n_periods - 1, index, :]).any(axis=0),
        ]

        current_policy_function[state] = partial(
            interpolate_policy, policy=final_policy,
        )

        current_value_function[state] = partial(
            utility_functions["utility"], state=state, params=params
        )

    # Start backwards induction from second to last period (T - 1)
    for period in range(n_periods - 2, -1, -1):

        # Update and reset dictionaries
        next_period_policy_function = current_policy_function
        next_period_value_function = current_value_function

        current_policy_function, current_value_function = dict(), dict()

        for index, state in enumerate(choice_range):
            current_policy, current_value, expected_value = do_egm_step(
                period,
                state,
                params=params,
                options=options,
                exogenous_grid=exogenous_grid,
                utility_functions=utility_functions,
                compute_expected_value=compute_expected_value,
                next_period_policy_function=next_period_policy_function,
                next_period_value_function=next_period_value_function,
            )

            if state >= 1 and n_choices > 1:
                current_policy, current_value = do_upper_envelope_step(
                    current_policy,
                    current_value,
                    expected_value=expected_value,
                    params=params,
                    options=options,
                    compute_utility=utility_functions["utility"],
                )
            else:
                pass

            current_value_function[state] = partial(
                interpolate_value,
                value=current_value,
                state=state,
                params=params,
                utility_func=utility_functions["utility"],
            )

            current_policy_function[state] = partial(
                interpolate_policy, policy=current_policy,
            )

            # Store
            policy_arr[period, index, :, : current_policy.shape[1]] = current_policy
            value_arr[period, index, :, : current_value.shape[1]] = current_value

    return policy_arr, value_arr


def interpolate_policy(flat_wealth: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """
    """
    policy = policy[:, ~np.isnan(policy).any(axis=0)]
    policy_interp = np.empty(flat_wealth.shape)

    interpolation_func = interpolate.interp1d(
        x=policy[0, :],
        y=policy[1, :],
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )

    policy_interp = interpolation_func(flat_wealth)

    return policy_interp


def interpolate_value(
    wealth: np.ndarray,
    value: np.ndarray,
    state: int,
    params: pd.DataFrame,
    utility_func: Callable,
) -> np.ndarray:
    """
    
    """
    value = value[:, ~np.isnan(value).any(axis=0)]
    value_interp = np.empty(wealth.shape)

    # Mark credit constrained region
    constrained_region = wealth < value[0, 1]

    # Calculate t+1 value function in constrained region using
    # the analytical part
    value_interp[constrained_region] = _get_value_constrained(
        wealth[constrained_region],
        next_period_value=value[1, 0],
        state=state,
        params=params,
        utility_func=utility_func,
    )

    # Calculate t+1 value function in non-constrained region
    # via inter- and extrapolation
    interpolation_func = interpolate.interp1d(
        x=value[0, :],  # endogenous wealth grid
        y=value[1, :],  # value_function
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )
    value_interp[~constrained_region] = interpolation_func(wealth[~constrained_region])

    return value_interp


def _get_value_constrained(
    wealth: np.ndarray,
    next_period_value: np.ndarray,
    state: int,
    params: pd.DataFrame,
    utility_func: Callable,
) -> np.ndarray:
    """"
    
    """
    beta = params.loc[("beta", "beta"), "value"]

    utility = utility_func(wealth, state, params)
    value_constrained = utility + beta * next_period_value

    return value_constrained


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
    compute_utility: Callable,
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
        value_func (callable): The agent's value function in the final period.
    Returns:
        (tuple): Tuple containing
        - policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies with the solution for the final 
            period included.
        - value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions with the solution for the final period
            included.
    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else range(n_choices)

    # In last period, nothing is saved for the next period (since there is none).
    # Hence, everything is consumed, c_T(M, d) = M
    end_grid = savings_grid.shape[0] + 1
    for state_index, state in enumerate(choice_range):
        policy[n_periods - 1, state_index, 0, 1:end_grid] = copy.deepcopy(
            savings_grid
        )  # M
        policy[n_periods - 1, state_index, 1, 1:end_grid] = copy.deepcopy(
            policy[n_periods - 1, state_index, 0, 1:end_grid]
        )  # c(M, d)
        policy[n_periods - 1, state_index, 0, 0] = 0
        policy[n_periods - 1, state_index, 1, 0] = 0

        value[n_periods - 1, state_index, 0, 2:end_grid] = compute_utility(
            policy[n_periods - 1, state_index, 0, 2:end_grid], state, params
        )
        value[n_periods - 1, state_index][1, 2:end_grid] = compute_utility(
            policy[n_periods - 1, state_index, 1, 2:end_grid], state, params
        )
        value[n_periods - 1, state_index, 0, 0] = 0
        value[n_periods - 1, state_index, :, 2] = 0

    return policy, value


def _create_multi_dim_arrays(
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

    policy_arr = np.empty((n_periods, n_choices, 2, int(1.1 * n_grid_wealth)))
    value_arr = np.empty((n_periods, n_choices, 2, int(1.1 * n_grid_wealth)))
    policy_arr[:] = np.nan
    value_arr[:] = np.nan

    return policy_arr, value_arr
