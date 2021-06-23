"""Model specific utility, wealth, and value functions."""
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import interpolate


def utility_func_crra(
    current_consumption: np.ndarray, params: pd.DataFrame
) -> np.ndarray:
    """Computes the agent's current utility based on a CRRA utility function.

    The utility is calculated at each point in the wealth grid M.

    Args:
        current_consumption (np.ndarray): Level of the agent's consumption in the
            current period.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (np.ndarray): Array of agent's utility in the current period
            with length (n_grid_wealth,).
    """
    theta = params.loc[("utility_function", "theta"), "value"]

    if theta == 1:
        utility = np.log(current_consumption)
    else:
        utility = (current_consumption ** (1 - theta) - 1) / (1 - theta)

    return utility


def marginal_utility_crra(
    current_consumption: np.ndarray, params: pd.DataFrame
) -> np.ndarray:
    """Computes marginal utility of CRRA utility function.

    Calculated at each point in the wealth grid M.

    Args:
        current_consumption (np.ndarray): Level of the agent's consumption in the
            curent period.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
            Relevant here is the CRRA coefficient theta.

    Returns:
        marginal_utility (np.ndarray): Marginal utility of CRRA consumption
            function with length (n_grid_wealth,).
    """
    theta = params.loc[("utility_function", "theta"), "value"]
    marginal_utility = current_consumption ** (-theta)

    return marginal_utility


def inverse_marginal_utility_crra(
    marginal_utility: np.ndarray,
    params: pd.DataFrame,
) -> np.ndarray:
    """Computes the inverse marginal utility of a CRRA utility function.

    Calculated at each point in the wealth grid M.

    Args:
        marginal_utility (np.ndarray): Level of marginal CRRA utility.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
    Returns:
        inverse_marginal_utility(np.ndarray): Inverse of the marginal utility of
            a CRRA consumption function.
    """
    theta = params.loc[("utility_function", "theta"), "value"]
    inverse_marginal_utility = marginal_utility ** (-1 / theta)

    return inverse_marginal_utility


def compute_current_income(
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


def compute_next_period_wealth_matrix(
    period: int,
    state: int,
    savings: float,
    quad_points: float,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> float:
    """Computes all possible levels of next period wealth M_(t+1)

    Args:
        period (int): Current period t.
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        savings_grid (np.ndarray): Array of length n_grid_wealth denoting the
            exogenous savings grid.
        quad_points (np.ndarray): Array of length n_quad_stochastic
            containing (normally distributed) stochastic components.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
        next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
    """
    r = params.loc[("assets", "interest_rate"), "value"]
    sigma = params.loc[("shocks", "sigma"), "value"]

    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]

    shocks = quad_points * sigma
    next_period_income = compute_current_income(period + 1, shocks, params, options)

    next_period_wealth = (
        np.full(
            (n_grid_wealth, n_quad_stochastic),
            next_period_income * state,
        ).T
        + np.full((n_quad_stochastic, n_grid_wealth), savings * (1 + r))
    )

    return next_period_wealth


def compute_marginal_wealth_matrix(
    params: pd.DataFrame, options: Dict[str, int]
) -> np.ndarray:
    """Computes marginal next period wealth.

    Args:
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
        marginal_wealth (np.ndarray):  Array of all possible next period
            marginal wealths. Shape (n_quad_stochastic, n_grid_wealth).
    """
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]
    r = params.loc[("assets", "interest_rate"), "value"]

    marginal_wealth = np.full((n_quad_stochastic, n_grid_wealth), (1 + r))

    return marginal_wealth


def compute_value_function(
    value: np.ndarray,
    next_period_wealth_matrix: np.ndarray,
    next_period: int,
    state: int,
    utility_function: Callable,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> np.ndarray:
    """Computes the value function of the next period t+1.

    Take into account credit-constrained regions.
    Use interpolation in non-constrained region and apply extrapolation
    where the observed wealth exceeds the maximum wealth level.

    Args:
        value (np.ndarray): Multi-dimensional array storing values of value
            function over all time periods.
            Shape (n_periods, n_choices, 2, n_grid_wealth + 1).
        next_period_wealth_matrix (np.ndarray): Array of of all possible next
            period wealths. Shape (n_quad_stochastic, n_grid_wealth).
        next_period (int): Next period, t+1.
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
        uility_func (callable): Utility function.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
        value_function (np.ndarray): Value function.
    """
    delta = params.loc[("delta", "delta"), "value"]
    beta = params.loc[("beta", "beta"), "value"]

    # If only one state, i.e. no discrete choices to make,
    # set state index to 0
    state_index = 0 if options["n_discrete_choices"] < 2 else state

    next_period_wealth = next_period_wealth_matrix.flatten("F")

    value_function = np.full(next_period_wealth.shape, np.nan)

    # Mark credit constrained region
    # Credit constrain lies between first (M_{t+1} = 0)
    # and second point (A_{t+1} = 0)
    # page number? --> Look up in paper for economic intuition!!

    constrained_region = next_period_wealth < value[next_period, state_index, 0, 1]
    # Last dim denotes grid point j+1

    # Calculate t+1 value function in constrained region
    value_function[constrained_region] = (
        utility_function(next_period_wealth[constrained_region], params)
        - state * delta
        + beta * value[next_period, state_index, 1, 0]
    )

    # Calculate t+1 value function in non-constrained region
    # via inter- and extrapolation
    value_function_interp = interpolate.interp1d(
        x=value[next_period, state_index, 0, :],  # endogenous wealth grid
        y=value[next_period, state_index, 1, :],  # value_function
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )
    value_function[~constrained_region] = value_function_interp(
        next_period_wealth[~constrained_region]
    )

    return value_function
