from jax import numpy as jnp


# =====================================================================================
# Utility functions
# =====================================================================================


def flow_utility(consumption, choice, params):
    return consumption ** (1 - params["rho"]) / (1 - params["rho"]) - params[
        "delta"
    ] * (1 - choice)


def marginal_utility(consumption, params):
    return consumption ** (-params["rho"])


def inverse_marginal_utility(marginal_utility, params):
    return marginal_utility ** (-1 / params["rho"])


# =====================================================================================
# Exogenous LTC process
# =====================================================================================


def prob_exog_ltc(
    period,
    ltc,
    params,
):
    prob_ltc = (ltc == 0) * (
        params["ltc_prob_constant"] + period * params["ltc_prob_age"]
    ) + (ltc == 1)
    prob_no_ltc = 1 - prob_ltc

    return jnp.array([prob_no_ltc, prob_ltc])


def budget_dcegm_exog_ltc(
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


# =====================================================================================
# Exogenous LTC and job offer processes
# =====================================================================================


def prob_exog_job_offer(
    job_offer,
    params,
):
    prob_job_offer = (job_offer == 0) * params["job_offer_constant"] + (
        job_offer == 1
    ) * (params["job_offer_constant"] + params["job_offer_type_two"])
    prob_no_job_offer = 1 - prob_job_offer

    return jnp.array([prob_no_job_offer, prob_job_offer])


def budget_dcegm_exog_ltc_and_job_offer(
    lagged_choice,
    ltc,
    savings_end_of_previous_period,
    income_shock_previous_period,
    options,
    params,
):
    # lagged_job_offer = jnp.abs(state[-1] - 2) * (state[-1] > 0) * state[0]  # [1, 3]
    ltc_patient = ltc == 1  # [2, 3]

    resource = (
        (1 + params["interest_rate"]) * savings_end_of_previous_period
        + (params["wage_avg"] + income_shock_previous_period)
        * (1 - lagged_choice)  # if worked last period
        - ltc_patient * params["ltc_cost"]
    )
    return jnp.maximum(resource, 0.5)


# =====================================================================================
# Exogenous healht processes
# =====================================================================================


def prob_exog_health(health, params):
    prob_good_health = (health == 0) * 0.7 + (health == 1) * 0.3 + (health == 2) * 0.2
    prob_medium_health = (health == 0) * 0.2 + (health == 1) * 0.5 + (health == 2) * 0.2
    prob_bad_health = (health == 0) * 0.1 + (health == 1) * 0.2 + (health == 2) * 0.6

    return jnp.array([prob_good_health, prob_medium_health, prob_bad_health])
