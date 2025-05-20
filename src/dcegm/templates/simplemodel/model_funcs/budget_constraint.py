from typing import Any, Dict

import jax
import jax.numpy as jnp


def budget_constraint(
    period: int,
    lagged_choice: int,
    asset_end_of_previous_period: float,
    income_shock_previous_period: float,
    model_specs: Dict[str, Any],
    params: Dict[str, float],
) -> float:
    """Compute possible current beginning of period resources.

    Given the savings grid of the previous period (t -1) and the current
    state vector in t, including the agent's lagged discrete choice made in t-1.

    Args:
        state_beginning_of_period (np.ndarray): 1d array of shape (n_state_variables,)
            denoting the current child state.
        savings_end_of_previous_period (float): One point on the exogenous savings grid
            carried over from the previous period.
        income_shock_pervious_period (float): Stochastic shock on labor income;
            may or may not be normally distributed. This float represents one
            particular realization of the income_shock_draws carried over from the
            previous period.
        params (dict): Dictionary containing model parameters.
        model_specs (dict): model_specs dictionary.

    Returns:
        (float): The beginning of period wealth in t.

    """
    # Calculate stochastic labor income
    income_from_previous_period = _calc_stochastic_income(
        period=period,
        lagged_choice=lagged_choice,
        wage_shock=income_shock_previous_period,
        min_age=model_specs["min_age"],
        constant=params["constant"],
        exp=params["exp"],
        exp_squared=params["exp_squared"],
    )

    wealth_beginning_of_period = (
        income_from_previous_period
        + (1 + params["interest_rate"]) * asset_end_of_previous_period
    )

    # Retirement safety net, only in retirement model, but we require to have it always
    # as a parameter
    wealth_beginning_of_period = jnp.maximum(
        wealth_beginning_of_period, params["consumption_floor"]
    )

    return wealth_beginning_of_period


@jax.jit
def _calc_stochastic_income(
    period: int,
    lagged_choice: int,
    wage_shock: float,
    min_age: int,
    constant: float,
    exp: float,
    exp_squared: float,
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
        state (jnp.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        wage_shock (float): Stochastic shock on labor income;
            may or may not be normally distributed. This float represents one
            particular realization of the income_shock_draws carried over from
            the previous period.
        params (dict): Dictionary containing model parameters.
            Relevant here are the coefficients of the wage equation.
        model_specs (dict): model_specs dictionary.

    Returns:
        stochastic_income (float): The potential end of period income. It consists of a
            deterministic component, i.e. age-dependent labor income,
            and a stochastic shock.

    """
    # For simplicity, assume current_age - min_age = experience
    age = period + min_age

    # Determinisctic component of income depending on experience:
    # constant + alpha_1 * age + alpha_2 * age**2
    exp_coeffs = jnp.array([constant, exp, exp_squared])
    labor_income = exp_coeffs @ (age ** jnp.arange(len(exp_coeffs)))
    working_income = jnp.exp(labor_income + wage_shock)

    return (1 - lagged_choice) * working_income
