import numpy as np

from dcegm.toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    marginal_utility_crra,
    utility_crra,
)


def prob_long_term_care_patient(params, lagged_bad_health, bad_health):
    p = params["ltc_prob_constant"]

    if lagged_bad_health == bad_health == 0:
        prob = 1 - p  # 0.7
    elif (lagged_bad_health == 0) and (bad_health == 1):
        prob = p  # 0.3
    elif lagged_bad_health == 1 and bad_health == 0:
        prob = 0
    elif lagged_bad_health == bad_health == 1:
        prob = 1

    return prob


def wage(nu, params):
    wage = params["wage_avg"] + nu
    return wage


def choice_prob_retirement(consumption, choice, params):
    v = utility_crra(consumption=consumption, params=params, choice=choice)
    v_0 = utility_crra(consumption=consumption, params=params, choice=0)
    v_1 = utility_crra(consumption=consumption, params=params, choice=1)
    choice_prob = np.exp(v) / (np.exp(v_0) + np.exp(v_1))
    return choice_prob


# =====================================================================================
# Exogenous LTC process
# =====================================================================================


def budget_exog_ltc(
    lagged_resources,
    lagged_consumption,
    lagged_retirement_choice,
    wage,
    bad_health,
    params,
):
    interest_factor = 1 + params["interest_rate"]
    health_costs = params["ltc_cost"]
    resources = (
        interest_factor * (lagged_resources - lagged_consumption)
        + wage * (1 - lagged_retirement_choice)
        - bad_health * health_costs
    ).clip(min=0.5)
    return resources


def marginal_utility_weighted_exog_ltc(
    init_cond, params, retirement_choice_1, nu, consumption
):
    """Return the expected marginal utility for one realization of the wage shock."""
    budget_1 = init_cond["assets_end_of_period"]
    ltc_state_1 = init_cond["bad_health"]

    weighted_marginal = 0
    for ltc_state_2 in (0, 1):
        for retirement_choice_2 in (0, 1):
            budget_2 = budget_exog_ltc(
                budget_1,
                consumption,
                retirement_choice_1,
                wage(nu, params),
                ltc_state_2,
                params,
            )

            marginal_util = marginal_utility_crra(consumption=budget_2, params=params)
            choice_prob = choice_prob_retirement(
                consumption=budget_2, choice=retirement_choice_2, params=params
            )

            ltc_prob = prob_long_term_care_patient(params, ltc_state_1, ltc_state_2)

            weighted_marginal += choice_prob * ltc_prob * marginal_util

    return weighted_marginal


def euler_rhs_exog_ltc(
    init_cond, params, draws, weights, retirement_choice_1, consumption
):
    discount_factor = params["discount_factor"]
    interest_factor = 1 + params["interest_rate"]

    rhs = 0
    for index_draw, draw in enumerate(draws):
        marg_util_draw = marginal_utility_weighted_exog_ltc(
            init_cond, params, retirement_choice_1, draw, consumption
        )
        rhs += weights[index_draw] * marg_util_draw
    return rhs * discount_factor * interest_factor


# =====================================================================================
# Exogenous LTC and job offer processes
# =====================================================================================


def prob_job_offer(params, lagged_job_offer, job_offer):
    # p = params["job_offer_prob"]

    if (lagged_job_offer == 0) and (job_offer == 1):
        prob = 0.5
    elif lagged_job_offer == job_offer == 0:
        prob = 0.5
    elif lagged_job_offer == 1 and job_offer == 0:
        prob = 0.1
    elif lagged_job_offer == job_offer == 1:
        prob = 0.9

    return prob


def budget_exog_ltc_and_job_offer(
    lagged_resources,
    lagged_consumption,
    lagged_retirement_choice,
    wage,
    bad_health,
    lagged_job_offer,
    params,
):
    interest_factor = 1 + params["interest_rate"]
    health_costs = params["ltc_cost"]
    resources = (
        interest_factor * (lagged_resources - lagged_consumption)
        + wage * lagged_job_offer * (1 - lagged_retirement_choice)
        - bad_health * health_costs
    ).clip(min=0.5)
    return resources


def marginal_utility_weighted_exog_ltc_and_job_offer(
    init_cond, params, retirement_choice_1, nu, consumption
):
    """Return the expected marginal utility for one realization of the wage shock."""
    budget_1 = init_cond["assets_end_of_period"]
    ltc_state_1 = init_cond["bad_health"]
    job_state_1 = init_cond["job_offer"]

    weighted_marginal = 0
    for ltc_state_2 in (0, 1):
        for job_state_2 in (0, 1):
            for retirement_choice_2 in (0, 1):
                budget_2 = budget_exog_ltc_and_job_offer(
                    budget_1,
                    consumption,
                    retirement_choice_1,
                    wage(nu, params),
                    ltc_state_2,
                    job_state_1,
                    params,
                )

                marginal_util = marginal_utility_crra(
                    consumption=budget_2, params=params
                )
                choice_prob = choice_prob_retirement(
                    consumption=budget_2, choice=retirement_choice_2, params=params
                )

                ltc_prob = prob_long_term_care_patient(params, ltc_state_1, ltc_state_2)
                job_offer_prob = prob_job_offer(params, job_state_1, job_state_2)

                weighted_marginal += (
                    choice_prob * ltc_prob * job_offer_prob * marginal_util
                )

    return weighted_marginal


def euler_rhs_exog_ltc_and_job_offer(
    init_cond, params, draws, weights, retirement_choice_1, consumption
):
    discount_factor = params["discount_factor"]
    interest_factor = 1 + params["interest_rate"]

    rhs = 0
    for index_draw, draw in enumerate(draws):
        marg_util_draw = marginal_utility_weighted_exog_ltc_and_job_offer(
            init_cond, params, retirement_choice_1, draw, consumption
        )
        rhs += weights[index_draw] * marg_util_draw
    return rhs * discount_factor * interest_factor
