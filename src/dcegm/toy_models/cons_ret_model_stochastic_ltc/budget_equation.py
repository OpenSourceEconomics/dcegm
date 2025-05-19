from jax import numpy as jnp


def budget_equation_with_ltc(
    ltc,
    lagged_choice,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    resource = (
        (1 + params["interest_rate"]) * asset_end_of_previous_period
        + (params["wage_avg"] + income_shock_previous_period)
        * (1 - lagged_choice)  # if worked last period
        - ltc * params["ltc_cost"]
    )
    return jnp.maximum(resource, 0.5)
