from typing import Dict

import jax
import jax.numpy as jnp


def budget_constraint(
    beginning_of_period_state: jnp.ndarray,
    end_of_last_period_saving: float,
    last_period_income_shock: float,
    params: Dict[str, float],
    options: Dict[str, int],
) -> float:
    """Compute possible current beginning of period resources, given the savings grid of
    last period and the current state including the choice of last period.

    Args:
        beginning_of_period_state (np.ndarray): 1d array of shape (n_state_variables,)
            denoting the current child state.
        end_of_last_period_saving (float): Entry of exogenous savings grid.
        last_period_income_shock (float): Stochastic shock on labor income; may or may
            not be normally distributed. Entry of income_shock_draws.
        params (dict): Dictionary containing model parameters.
        options (dict): Options dictionary.

    Returns:
        beginning_period_wealth (float): The current beginning of period resources.

    """
    r = params["interest_rate"]

    # Calculate stochastic labor income
    income_from_last_period = _calc_stochastic_income(
        beginning_of_period_state,
        wage_shock=last_period_income_shock,
        params=params,
        options=options,
    )

    beginning_period_wealth = (
        income_from_last_period + (1 + r) * end_of_last_period_saving
    )

    # Retirement safety net, only in retirement model, but we require to have it always
    # as a parameter
    beginning_period_wealth = jnp.maximum(
        beginning_period_wealth, params["consumption_floor"]
    )

    return beginning_period_wealth


@jax.jit
def _calc_stochastic_income(
    state: jnp.ndarray,
    wage_shock: float,
    params: Dict[str, float],
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
        state (jnp.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        wage_shock (float): Stochastic shock on labor income; may or may not be normally
            distributed. Entry of income_shock_draws.
        params (dict): Dictionary containing model parameters.
            Relevant here are the coefficients of the wage equation.
        options (dict): Options dictionary.

    Returns:
        stochastic_income (float): The potential end of period income. It consists of a
            deterministic component, i.e. age-dependent labor income,
            and a stochastic shock.

    """
    # For simplicity, assume current_age - min_age = experience
    min_age = options["min_age"]
    age = state[0] + min_age

    # Determinisctic component of income depending on experience:
    # constant + alpha_1 * age + alpha_2 * age**2
    exp_coeffs = jnp.array([params["constant"], params["exp"], params["exp_squared"]])
    labor_income = exp_coeffs @ (age ** jnp.arange(len(exp_coeffs)))
    working_income = jnp.exp(labor_income + wage_shock)
    return (1 - state[1]) * working_income
