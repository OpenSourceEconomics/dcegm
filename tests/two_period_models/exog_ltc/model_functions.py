from jax import numpy as jnp


def budget_dcegm(
    ltc,
    lagged_choice,
    savings_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    resource = (
        (1 + params["interest_rate"]) * savings_end_of_previous_period
        + (params["wage_avg"] + income_shock_previous_period)
        * (1 - lagged_choice)  # if worked last period
        - ltc * params["ltc_cost"]
    )
    return jnp.maximum(resource, 0.5)


def func_exog_ltc(
    period,
    ltc,
    params,
):
    prob_ltc = (ltc == 0) * (
        params["ltc_prob_constant"] + period * params["ltc_prob_age"]
    ) + (ltc == 1)
    prob_no_ltc = 1 - prob_ltc

    return jnp.array([prob_no_ltc, prob_ltc])


def flow_util(consumption, choice, params):
    return consumption ** (1 - params["rho"]) / (1 - params["rho"]) - params[
        "delta"
    ] * (1 - choice)


def marginal_utility(consumption, params):
    return consumption ** (-params["rho"])


def inverse_marginal_utility(marginal_utility, params):
    return marginal_utility ** (-1 / params["rho"])
