from typing import Dict

import numpy as np
import pandas as pd


def budget_constraint(
    state: np.ndarray,
    savings_grid: np.ndarray,
    income_shock: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> np.ndarray:
    """Compute possible current beginning of period resources, given the savings grid of
    last period and the current state including the choice of last period.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            exogenous savings grid.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        income_shock (np.ndarray): 1d array of shape (n_quadrature_points,) containing
            a stochastic shock on labor income; may or may not be normally distributed.

    Returns:
        (np.ndarray): 2d array of shape (n_quad_stochastic, n_grid_wealth)
            containing all possible next period wealths.

    """
    r = params.loc[("assets", "interest_rate"), "value"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]

    # Calculate stochastic labor income
    _next_period_income = _calc_stochastic_income(
        state,
        wage_shock=income_shock,
        params=params,
        options=options,
    )
    income_matrix = np.repeat(_next_period_income[:, np.newaxis], n_grid_wealth, 1)
    savings_matrix = np.repeat(savings_grid[np.newaxis, :], n_quad_stochastic, 0) * (
        1 + r
    )
    matrix_next_period_wealth = income_matrix + savings_matrix

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

    return matrix_next_period_wealth


def _calc_stochastic_income(
    child_state: np.ndarray,
    wage_shock: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
) -> float:
    """Computes the current level of deterministic and stochastic income.

    Note that income is paid at the end of the current period, i.e. after
    the (potential) labor supply choice has been made. This is equivalent to
    allowing income to be dependent on a lagged choice of labor supply.
    The agent starts working in period t = 0.
    Relevant for the wage equation (deterministic income) are age-dependent
    coefficients of work experience:
    labor_income = constant + alpha_1 * age + alpha_2 * age**2
    They include a constant as well as two coefficients on age and age squared,
    respectively. Note that the last one (alpha_2) typically has a negative sign.

    Args:
        child_state (np.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        wage_shock (np.ndarray): 1d array of shape (n_quadrature_points,) containing
            a stochastic shock on labor income; may or may not be normally distributed.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
            Relevant here are the coefficients of the wage equation.
        options (dict): Options dictionary.

    Returns:
        stochastic_income (np.ndarray): 1d array of shape (n_quad_points,) containing
            potential end of period incomes. It consists of a deterministic component,
            i.e. age-dependent labor income, and a stochastic shock.

    """
    if child_state[1] == 0:  # working
        # For simplicity, assume current_age - min_age = experience
        min_age = options["min_age"]
        age = child_state[0] + min_age

        # Determinisctic component of income depending on experience:
        # constant + alpha_1 * age + alpha_2 * age**2
        exp_coeffs = np.asarray(params.loc["wage", "value"])
        labor_income = exp_coeffs @ (age ** np.arange(len(exp_coeffs)))

        stochastic_income = np.exp(labor_income + wage_shock)

    elif child_state[1] == 1:  # retired
        stochastic_income = np.zeros_like(wage_shock)

    return stochastic_income


def marginal_wealth(child_state, params, options):  # noqa: U100
    """Calculate next periods marginal wealth.

    Args:
        child_state (np.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.

    Returns:
         (np.ndarray): 2d array of shape (n_quad_stochastic, n_grid_wealth)
            containing all possible next marginal period wealths.

    """
    r = params.loc[("assets", "interest_rate"), "value"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]

    out = np.full((n_quad_stochastic, n_grid_wealth), (1 + r))

    return out
