import jax.numpy as jnp


def budget_constraint_exp(
    lagged_choice,
    experience,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
):

    working = lagged_choice == 0

    income_from_previous_period = _calc_stochastic_income(
        experience=experience,
        wage_shock=income_shock_previous_period,
        params=params,
    )

    wealth_beginning_of_period = (
        income_from_previous_period * working
        + (1 + params["interest_rate"]) * asset_end_of_previous_period
    )

    # Retirement safety net, only in retirement model, but we require to have it always
    # as a parameter
    return jnp.maximum(wealth_beginning_of_period, params["consumption_floor"])


def _calc_stochastic_income(
    experience,
    wage_shock,
    params,
):

    labor_income = (
        params["constant"]
        + params["exp"] * experience
        + params["exp_squared"] * experience**2
    )

    return jnp.exp(labor_income + wage_shock)
