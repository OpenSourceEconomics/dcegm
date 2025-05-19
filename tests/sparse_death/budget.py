import jax.numpy as jnp


def budget_constraint_exp(
    lagged_choice,
    experience,
    already_retired,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
    model_specs,
):

    # If unemployed, then it will just be the consumption floor
    working = lagged_choice == 2
    retired = lagged_choice == 0

    # Check if fresh retired. If so we give a bonus
    fresh_retired = retired * (1 - already_retired)
    replacement_rate = 0.48 + fresh_retired * model_specs["fresh_bonus"]

    # Scale experience
    exp_years = experience * model_specs["exp_scale"]

    income_from_previous_period = _calc_stochastic_income(
        experience=exp_years,
        wage_shock=income_shock_previous_period,
        params=params,
    )

    wealth_beginning_of_period = (
        income_from_previous_period * working
        + income_from_previous_period
        * replacement_rate
        * retired  # 0.48 is the replacement rate
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
