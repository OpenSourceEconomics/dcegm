"""Model specific utility, wealth, and value functions."""
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd


def utility_func_crra(
    consumption: np.ndarray, choice: int, params: pd.DataFrame
) -> np.ndarray:
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        consumption (np.ndarray): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (np.ndarray): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).
    """
    theta = params.loc[("utility_function", "theta"), "value"]
    delta = params.loc[("delta", "delta"), "value"]

    if theta == 1:
        utility_consumption = np.log(consumption)
    else:
        utility_consumption = (consumption ** (1 - theta) - 1) / (1 - theta)

    utility = utility_consumption - (1 - choice) * delta

    return utility


def marginal_utility_crra(consumption: np.ndarray, params: pd.DataFrame) -> np.ndarray:
    """Computes marginal utility of CRRA utility function.

    Args:
        consumption (np.ndarray): Level of the agent's consumption.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
            Relevant here is the CRRA coefficient theta.

    Returns:
        marginal_utility (np.ndarray): Marginal utility of CRRA consumption
            function. Array of shape (n_quad_stochastic * n_grid_wealth,).
    """
    theta = params.loc[("utility_function", "theta"), "value"]
    marginal_utility = consumption ** (-theta)

    return marginal_utility


def inverse_marginal_utility_crra(
    marginal_utility: np.ndarray,
    params: pd.DataFrame,
) -> np.ndarray:
    """Computes the inverse marginal utility of a CRRA utility function.

    Args:
        marginal_utility (np.ndarray): Level of marginal CRRA utility.
            Array of shape (n_grid_wealth,).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].

    Returns:
        inverse_marginal_utility(np.ndarray): Inverse of the marginal utility of
            a CRRA consumption function. Array of shape (n_grid_wealth,).
    """
    theta = params.loc[("utility_function", "theta"), "value"]
    beta = params.loc[("beta", "beta"), "value"]

    inverse_marginal_utility = (marginal_utility * beta) ** (-1 / theta)

    return inverse_marginal_utility


def budget_constraint(
    state,
    savings: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    income_shocks: Callable,
) -> np.ndarray:
    """Compute possible current beginning of period resources, given the savings grid of
    last period and the current state including the choice of last period.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        savings (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            exogenous savings grid.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        wage_shock (float): Stochastic shock on labor income, which may or may not
            be normally distributed.

    Returns:
        (np.ndarray): 2d array of shape (n_quad_stochastic, n_grid_wealth)
            containing all possible next period wealths.
    """
    r = params.loc[("assets", "interest_rate"), "value"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]

    # Calculate stochastic labor income
    _next_period_income = calc_stochastic_income(
        state,
        wage_shock=income_shocks,
        params=params,
        options=options,
    )
    income_matrix = np.repeat(_next_period_income[:, np.newaxis], n_grid_wealth, 1)
    savings_matrix = np.full((n_quad_stochastic, n_grid_wealth), savings * (1 + r))

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


def calc_stochastic_income(
    child_state: int,
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
        wage_shock (float): Stochastic shock on labor income, which may or may not
            be normally distributed.
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


def calc_next_period_marginal_wealth(state, params, options):
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


def solve_final_period(
    states: np.ndarray,
    savings_grid: np.ndarray,
    *,
    options: Dict[str, int],
    compute_utility: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes solution to final period for policy and value function.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        state_space (np.ndarray): Collection of all possible states.
        indexer (np.ndarray): Indexer object, that maps states to indexes.
        savings_grid (np.ndarray): Array of shape (n_wealth_grid,) denoting the
            exogenous savings grid.
        options (dict): Options dictionary.
        compute_utility (callable): Function for computation of agent's utility.

    Returns:
        (tuple): Tuple containing

        - policy (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific policy function; of shape
            [n_states, n_discrete_choices, 2, 1.1 * (n_grid_wealth + 1)].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each state and each discrete choice.
        - value (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_states, n_discrete_choices, 2, 1.1 * (n_grid_wealth + 1)].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each state and each discrete choice.
    """
    n_choices = options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else range(n_choices)
    n_states = states.shape[0]

    policy_final = np.empty(
        (n_states, n_choices, 2, int(1.1 * (len(savings_grid) + 1)))
    )
    value_final = np.empty((n_states, n_choices, 2, int(1.1 * (len(savings_grid) + 1))))
    policy_final[:] = np.nan
    value_final[:] = np.nan

    end_grid = len(savings_grid) + 1

    # In last period, nothing is saved for the next period (since there is none).
    # Hence, everything is consumed, c_T(M, d) = M
    for state_index in range(n_states):

        for index, choice in enumerate(choice_range):
            policy_final[state_index, index, :, 0] = 0
            policy_final[state_index, index, 0, 1:end_grid] = savings_grid  # M
            policy_final[state_index, index, 1, 1:end_grid] = savings_grid  # c(M, d)

            value_final[state_index, index, :, :2] = 0
            value_final[state_index, index, 0, 1:end_grid] = savings_grid

            # Start with second entry of savings grid to avaid taking the log of 0
            # (the first entry) when computing utility
            value_final[state_index, index, 1, 2:end_grid] = compute_utility(
                savings_grid[1:], choice
            )
    return policy_final, value_final
