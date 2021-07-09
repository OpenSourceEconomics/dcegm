"""Implementation of the EGM algorithm."""
import copy
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy import interpolate


def do_egm_step(
    period: int,
    state: int,
    policy: np.ndarray,
    value: np.ndarray,
    *,
    params: pd.DataFrame,
    options: Dict[str, int],
    exogenous_grid: Dict[str, np.ndarray],
    utility_functions: Dict[str, callable],
    value_functions: Dict[str, callable],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Runs the Endogenous-Grid-Method Algorithm (EGM step).

    Args:
        period (int): Current period t.
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            concurrent local optima for consumption. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1.
            Position [0, :] of the arrays contain the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the (consumption) policy 
            function c(M, d), for each time period and each discrete choice. 
        value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*], where 
            *n_endog_wealth_grid* is of variable length depending on the number of 
            kinks and non-concave regions in the subsequent period (t + 1), which
            propagate back to the current period t. The arrays have 
            shape [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1.
            Position [0, :] of the array contains the endogenous grid over wealth M, 
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice. 
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        exogenous_grid (Dict[str, np.ndarray]): Dictionary containing the
            exogenous grids of (i) savings (array of shape (n_grid_wealth, ))
            (ii) quadrature points (array of shape (n_quad_stochastic, )) and
            (iii) associated quadrature weights (also an array of shape
            (n_quad_stochastic, )).
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of (i) utility, (ii) inverse marginal utility, 
            and (iii) next period marginal utility.
        value_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of the agent's (i) value function before
            the final period, (ii) value function in the final period, 
            and (iii) the expected value.

    Returns:
        (tuple) Tuple containing
        
        - policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*]. 
        - value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*].
    """
    # 0) Preliminaries
    # Matrices of all possible next period wealths and marginal wealths
    (
        matrix_next_period_wealth,
        next_period_marginal_wealth,
    ) = get_next_period_wealth_matrices(
        period,
        state,
        params=params,
        options=options,
        savings=exogenous_grid["savings"],
        quad_points=exogenous_grid["quadrature_points"],
    )
    next_period_value = get_next_period_value(
        period,
        value,
        matrix_next_period_wealth=matrix_next_period_wealth,
        params=params,
        options=options,
        value_functions=value_functions,
    )

    # 1) Endogenous wealth grid & current period consumption
    endog_wealth_grid, current_period_consumption = map_exog_to_endog_grid(
        period,
        state,
        policy,
        matrix_next_period_wealth=matrix_next_period_wealth,
        next_period_marginal_wealth=next_period_marginal_wealth,
        next_period_value=next_period_value,
        params=params,
        options=options,
        exogenous_grid=exogenous_grid,
        utility_functions=utility_functions,
    )

    # 2) Value function
    expected_value, current_period_value = get_expected_and_current_period_value(
        state,
        next_period_value=next_period_value,
        matrix_next_period_wealth=matrix_next_period_wealth,
        current_period_consumption=current_period_consumption,
        quad_weights=exogenous_grid["quadrature_weights"],
        params=params,
        options=options,
        value_functions=value_functions,
    )

    # 3) Update policy and value function
    # If no discrete alternatives; only one state, i.e. one column with index = 0
    state_index = 0 if options["n_discrete_choices"] < 2 else state

    policy[period][state_index][0, 1:] = endog_wealth_grid
    policy[period][state_index][1, 1:] = current_period_consumption

    value[period][state_index][0, 1:] = endog_wealth_grid
    value[period][state_index][1, 1:] = current_period_value
    value[period][state_index][1, 0] = expected_value[0]

    return policy, value, expected_value


def get_next_period_wealth_matrices(
    period: int,
    state: int,
    savings: np.ndarray,
    quad_points: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes all possible levels of next period (marginal) wealth M_(t+1).

    Args:
        period (int): Current period t.
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        savings (np.ndarray): Array of shape (n_grid_wealth,) containing the
            exogenous savings grid.
        quad_points (np.ndarray): Array of shape (n_quad_stochastic,)
            containing (normally distributed) stochastic income components,
            which induce shocks to the wage equation.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
        (tuple): Tuple containing

        - matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        - next_period_marginal_wealth (np.ndarray): Array of all possible next period
            marginal wealths. Also of shape (n_quad_stochastic, n_grid_wealth).
    """
    r = params.loc[("assets", "interest_rate"), "value"]
    sigma = params.loc[("shocks", "sigma"), "value"]

    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]

    # Calculate stochastic labor income
    shocks = quad_points * sigma
    next_period_income = _calc_stochastic_income(period + 1, shocks, params, options)

    matrix_next_period_wealth = np.full(
        (n_grid_wealth, n_quad_stochastic), next_period_income * state,
    ).T + np.full((n_quad_stochastic, n_grid_wealth), savings * (1 + r))

    # Retirement safety net, only in retirement model
    consump_floor_index = ("assets", "consumption_floor")
    if (
        consump_floor_index in params.index
        or params.loc[consump_floor_index, "value"] > 0
    ):
        consump_floor = params.loc[consump_floor_index, "value"]
        matrix_next_period_wealth[
            matrix_next_period_wealth < consump_floor
        ] = consump_floor

    next_period_marginal_wealth = np.full((n_quad_stochastic, n_grid_wealth), (1 + r))

    return matrix_next_period_wealth, next_period_marginal_wealth


def get_next_period_value(
    period: int,
    value: List[np.ndarray],
    matrix_next_period_wealth: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    value_functions: Dict[str, callable],
) -> np.ndarray:
    """Computes the next-period (choice-specific) value function.

    Args:
        period (int): Current period t.
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
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        value_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of the agent's (i) value function before
            the final period, (ii) value function in the final period, 
            and (iii) the expected value.

    Returns:
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """
    n_periods, n_choices = options["n_periods"], options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else range(n_choices)

    compute_value_function_final_period = value_functions["final_period"]
    compute_value_function = value_functions["before_final_period"]

    next_period_value = np.empty(
        (
            n_choices,
            matrix_next_period_wealth.shape[0] * matrix_next_period_wealth.shape[1],
        )
    )

    for index, state in enumerate(choice_range):
        if period + 1 == n_periods - 1:  # Final period
            next_period_value[index, :] = compute_value_function_final_period(
                state, matrix_next_period_wealth, params
            )

        else:
            next_period_value[index, :] = _interpolate_value_function(
                state,
                actual_value=value[period + 1][index],
                matrix_next_period_wealth=matrix_next_period_wealth,
                params=params,
                compute_value_function=compute_value_function,
            )

    return next_period_value


def map_exog_to_endog_grid(
    period: int,
    state: int,
    policy: List[np.ndarray],
    matrix_next_period_wealth: np.ndarray,
    next_period_marginal_wealth: np.ndarray,
    next_period_value: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    exogenous_grid: Dict[str, np.ndarray],
    utility_functions: Dict[str, callable],
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes endogenous wealth grid and current period consumption.

    Args:
        period (int): Current period t.
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
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
            exogenous savings grid.
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_marginal_wealth (np.ndarray): Array of all possible next period
            marginal wealths. Also of shape (n_quad_stochastic, n_grid_wealth)
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        exogenous_grid (Dict[str, np.ndarray]): Dictionary containing the
            exogenous grids of (i) savings (array of shape (n_grid_wealth, ))
            (ii) quadrature points (array of shape (n_quad_stochastic, )) and
            (iii) associated quadrature weights (also an array of shape
            (n_quad_stochastic, )).
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of (i) utility, (ii) inverse marginal utility, 
            and (iii) next period marginal utility.

    Returns:
        (tuple): Tuple containing

        - endog_wealth_grid (np.ndarray): Endogenous wealth grid of shape
            (n_grid_wealth,). 
        - current_period_consumption (np.ndarray): Consumption in the current
            period. Array of shape (n_grid_wealth,).
    """
    savings_grid = exogenous_grid["savings"]
    quad_weights = exogenous_grid["quadrature_weights"]

    _compute_next_period_marginal_utility = utility_functions[
        "next_period_marginal_utility"
    ]

    next_period_consumption = _get_next_period_consumption(
        period, policy, matrix_next_period_wealth, options
    )

    next_period_marginal_utility = _compute_next_period_marginal_utility(
        state, next_period_consumption, next_period_value, params, options,
    )
    current_period_consumption = _get_current_period_consumption(
        next_period_marginal_utility,
        matrix_next_period_wealth,
        next_period_marginal_wealth,
        quad_weights,
        params,
        utility_functions["inverse_marginal_utility"],
    )

    endog_wealth_grid = savings_grid + current_period_consumption

    return endog_wealth_grid, current_period_consumption


def get_expected_and_current_period_value(
    state: int,
    next_period_value: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    current_period_consumption: np.ndarray,
    quad_weights: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    value_functions: Dict[str, callable],
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the expected (next period) value and the current period's value.
    
    Args:
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        current_period_consumption (np.ndarray): Consumption in the current
            period. Array of shape (n_grid_wealth,).
        quad_weights (np.ndarray): Weights associated with the quadrature points
            of shape (n_quad_stochastic,). Used for integration over the
            stochastic income component in the Euler equation.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        value_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of the agent's (i) value function before
            the final period, (ii) value function in the final period, 
            and (iii) the expected value.
    
    Returns:
        expected_value (np.ndarray): Expected value of next period. Array of
            shape (n_grid_wealth,).
        current_period_value (np.ndarray): Array of current period value
            function of shape (n_grid_wealth,).
    """
    compute_expected_value = value_functions["expected_value"]
    compute_value_function = value_functions["before_final_period"]

    expected_value = compute_expected_value(
        state,
        matrix_next_period_wealth,
        next_period_value=next_period_value,
        quad_weights=quad_weights,
        params=params,
        options=options,
    )
    current_period_value = compute_value_function(
        state,
        current_period_consumption,
        next_period_value=expected_value,
        params=params,
    )

    return expected_value, current_period_value


def _calc_stochastic_income(
    period: int, shock: float, params: pd.DataFrame, options
) -> float:
    """Computes the current level of deterministic and stochastic income.

    Note that income is paid at the end of the current period, i.e. after
    the (potential) labor supply choice has been made. This is equivalent to
    allowing income to be dependent on a lagged choice of labor supply.

    The agent starts working in period t = 0.
    Relevant for the wage equation (deterministic income) are age-dependent
    coefficients of work experience:

    labor_income = constant + alpha_1 * age + alpha_2 * age**2

    They include a constant as well as two coefficents on age and age squared,
    respectively. Note that the last one (alpha_2) typically has a negative sign.

    Args:
        period (int): Curent period t.
        shock (float): Stochastic shock on labor income, which may or may not
            be normally distributed.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
            Relevant here are the coefficients of the wage equation.
        options (dict): Options dictionary.

    Returns:
        stochastic_income (float): End of period income composed of a
            deterministic component, i.e. age-dependent labor income, and a
            stochastic shock.
    """
    # For simplicity, assume current_age - min_age = experience
    # TODO: Allow age and work experience to differ,
    # i.e. allow for unemployment spells
    min_age = options["min_age"]
    age = period + min_age

    # Determinisctic component of income depending on experience
    # labor_income = constant + alpha_1 * age + alpha_2 * age**2
    exp_coeffs = np.asarray(params.loc["wage", "value"])
    labor_income = exp_coeffs @ (age ** np.arange(len(exp_coeffs)))

    stochastic_income = np.exp(labor_income + shock)

    return stochastic_income


def _get_next_period_consumption(
    period: int,
    policy: List[np.ndarray],
    matrix_next_period_wealth: np.ndarray,
    options: Dict[str, int],
) -> np.ndarray:
    """Computes consumption in the next period via linear interpolation.

    Extrapolate lineary in wealth regions beyond the grid, i.e. larger
    than "max_wealth" specifiec in the ``params`` dictionary.

    Args:
        period (int): Current period t.
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
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        options (dict): Options dictionary.

    Returns:
        next_period_consumption_interp (np.ndarray): Array of next period
            consumption of shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """
    n_choices = options["n_discrete_choices"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]
    choice_range = [0] if n_choices < 2 else range(n_choices)

    next_period_consumption_interp = np.empty(
        (n_choices, n_quad_stochastic * n_grid_wealth)
    )

    for state_index in choice_range:
        actual_next_period_wealth = policy[period + 1][state_index][0, :]
        next_period_consumption = policy[period + 1][state_index][1, :]

        interpolation_func = interpolate.interp1d(
            actual_next_period_wealth,
            next_period_consumption,
            bounds_error=False,
            fill_value="extrapolate",
            kind="linear",
        )
        next_period_consumption_interp[state_index, :] = interpolation_func(
            matrix_next_period_wealth
        ).flatten("F")

    return next_period_consumption_interp


def _get_current_period_consumption(
    next_period_marginal_utility: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    matrix_marginal_wealth: np.ndarray,
    quad_weights: np.ndarray,
    params: pd.DataFrame,
    inv_marginal_utility_func: Callable,
) -> np.ndarray:
    """Computes consumption in the current period.

    Args:
        next_period_marginal_utility (np.ndarray): Array of next period's
            marginal utility of shape (n_quad_stochastic * n_grid_wealth,).
        matrix_next_period_wealth(np.ndarray): Array of all possible next
            period wealths. Shape (n_quad_stochastic, n_wealth_grid).
        next_period_marginal_wealth(np.ndarray): Array of marginal next period wealths.
            Shape (n_quad_stochastic, n_wealth_grid).
        quad_weights (np.ndarray): Weights associated with the quadrature points
            of shape (n_quad_stochastic,). Used for integration over the
            stochastic income component in the Euler equation.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        inv_marginal_utility_func (callable): Inverse of the marginal utility
            function.

    Returns:
        current_period_consumption (np.ndarray): Consumption in the current
            period. Array of shape (n_grid_wealth,).
    """
    beta = params.loc[("beta", "beta"), "value"]

    # RHS of Euler Eq., p. 337 IJRS (2017)
    # Integrate out uncertainty over stochastic income y
    rhs_euler = _calc_rhs_euler(
        next_period_marginal_utility,
        matrix_next_period_wealth,
        matrix_marginal_wealth,
        quad_weights,
    )

    current_period_consumption = inv_marginal_utility_func(beta * rhs_euler, params)

    return current_period_consumption


def _interpolate_value_function(
    state: int,
    actual_value: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    params: pd.DataFrame,
    compute_value_function: Callable,
) -> np.ndarray:
    """Computes the value function of the next period t+1.

    Take into account credit-constrained regions.
    Use interpolation in non-constrained region and apply extrapolation
    where the observed wealth exceeds the maximum wealth level.

    Args:
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        actual_value (np.ndarray): Actual next period value, obtained from
            previous (t + 1) run of the EGM and Upper Envelope Algorithms.
            Array of shape (2, *n_endog_wealth_grid*), where *n_endog_wealth_grid*
            is of variable length depending on the number of kinks and non-concave 
            regions in the next period.
        matrix_next_period_wealth (np.ndarray): Array of of all possible next
            period wealths. Shape (n_quad_stochastic, n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        uility_func (callable): Utility function.

    Returns:
        value_function (np.ndarray): Value function. Array of shape
            (n_quad_stochastic * n_grid_wealth,).
    """
    matrix_next_period_wealth = matrix_next_period_wealth.flatten("F")
    value_function = np.full(matrix_next_period_wealth.shape, np.nan)

    # Mark credit constrained region
    constrained_region = matrix_next_period_wealth < actual_value[0, 1]

    # Calculate t+1 value function in constrained region
    value_function[constrained_region] = compute_value_function(
        state,
        matrix_next_period_wealth[constrained_region],
        next_period_value=actual_value[1, 0],
        params=params,
    )

    # Calculate t+1 value function in non-constrained region
    # via inter- and extrapolation
    value_function_interp = interpolate.interp1d(
        x=actual_value[0, :],  # endogenous wealth grid
        y=actual_value[1, :],  # value_function
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )
    value_function[~constrained_region] = value_function_interp(
        matrix_next_period_wealth[~constrained_region]
    )

    return value_function


def _calc_rhs_euler(
    next_period_marginal_utility: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    next_period_marginal_wealth: np.ndarray,
    quad_weights: np.ndarray,
) -> np.ndarray:
    """Computes the right-hand side of the Euler equation, p. 337 IJRS (2017).

    Args:
        next_period_marginal_utility (np.ndarray): Array of next period's
            marginal utility of shape (n_quad_stochastic * n_grid_wealth,).
        matrix_next_period_wealth(np.ndarray): Array of all possible next
            period wealths. Shape (n_quad_stochastic, n_wealth_grid).
        next_period_marginal_wealth(np.ndarray): Array of marginal next period wealths.
            Shape (n_quad_stochastic, n_wealth_grid).
        quad_weights (np.ndarray): Weights associated with the quadrature points
            of shape (n_quad_stochastic,). Used for integration over the
            stochastic income component in the Euler equation.

    Returns:
        rhs_euler (np.ndarray): Right-hand side of the Euler equation.
            Shape (n_grid_wealth,).
    """
    next_period_marginal_utility = next_period_marginal_utility.reshape(
        matrix_next_period_wealth.shape, order="F"
    )

    rhs_euler = np.dot(
        quad_weights.T,
        np.multiply(next_period_marginal_utility, next_period_marginal_wealth),
    )

    return rhs_euler

