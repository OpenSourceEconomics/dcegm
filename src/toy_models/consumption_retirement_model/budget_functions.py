from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np


def budget_constraint(
    state: np.ndarray,
    saving: float,
    income_shock: float,
    params_dict: dict,
    options: Dict[str, int],
) -> float:
    """Compute possible current beginning of period resources, given the savings grid of
    last period and the current state including the choice of last period.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        saving (float): Entry of exogenous savings grid.
        params_dict (dict): Dictionary containing model parameters.
        options (dict): Options dictionary.
        income_shock (float): Stochastic shock on labor income; may or may not be
         normally distributed. Entry of income_shock_draws.

    Returns:
        (np.ndarray): 2d array of shape (n_quad_stochastic, n_grid_wealth)
            containing all possible next period wealths.

    """
    r = params_dict["interest_rate"]

    # Calculate stochastic labor income
    _next_period_income = _calc_stochastic_income(
        state,
        wage_shock=income_shock,
        params_dict=params_dict,
        options=options,
    )

    _next_period_wealth = _next_period_income + (1 + r) * saving

    # Retirement safety net, only in retirement model, but we require to have it always
    # as a parameter
    _next_period_wealth = jnp.maximum(
        _next_period_wealth, params_dict["consumption_floor"]
    )

    return _next_period_wealth


@jax.jit
def _calc_stochastic_income(
    child_state: np.ndarray,
    wage_shock: float,
    params_dict: dict,
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
        wage_shock (float): Stochastic shock on labor income; may or may not be normally
            distributed. Entry of income_shock_draws.
        params_dict (dict): Dictionary containing model parameters.
            Relevant here are the coefficients of the wage equation.
        options (dict): Options dictionary.

    Returns:
        stochastic_income (np.ndarray): 1d array of shape (n_quad_points,) containing
            potential end of period incomes. It consists of a deterministic component,
            i.e. age-dependent labor income, and a stochastic shock.

    """
    # For simplicity, assume current_age - min_age = experience
    min_age = options["min_age"]
    age = child_state[0] + min_age

    # Determinisctic component of income depending on experience:
    # constant + alpha_1 * age + alpha_2 * age**2
    exp_coeffs = jnp.array(
        [params_dict["constant"], params_dict["exp"], params_dict["exp_squared"]]
    )
    labor_income = exp_coeffs @ (age ** np.arange(len(exp_coeffs)))
    working_income = jnp.exp(labor_income + wage_shock)
    return (1 - child_state[1]) * working_income
