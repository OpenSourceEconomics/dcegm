from functools import partial

import numpy as np
import pandas as pd
import scipy.interpolate as scin
from dcegm.pre_processing import calc_current_value
from scipy.stats import norm
from toy_models.consumption_retirement_model.budget_functions import (
    _calc_stochastic_income,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

# from dcegm.retirement.ret import choice_probabilities
# from dcegm.egm import calc_choice_probability

# from dcegm.retirement.ret import income
# from dcegm.interpolation import interpolate_value


# from dcegm.retirement.ret import value_function
def calc_choice_probability(
    values: np.ndarray,
    taste_shock_scale: float,
) -> np.ndarray:
    """Calculate the next period probability of picking a given choice.

    Args:
        values (np.ndarray): Array containing choice-specific values of the
            value function. Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        taste_shock_scale (float): The taste shock scale parameter.

    Returns:
        (np.ndarray): Probability of picking the given choice next period.
            1d array of shape (n_quad_stochastic * n_grid_wealth,).

    """
    col_max = np.amax(values, axis=0)
    values_scaled = values - col_max

    # Eq. (15), p. 334 IJRS (2017)
    choice_prob = np.exp(values_scaled[1, :] / taste_shock_scale) / np.sum(
        np.exp(values_scaled / taste_shock_scale), axis=0
    )

    return choice_prob


def choice_probabilities(x, lambda_):
    """Calculate the probability of choosing work in t+1 for state worker given t+1
    value functions."""

    mx = np.amax(x, axis=0)
    mxx = x - mx
    res = np.exp(mxx[1, :] / lambda_) / np.sum(np.exp(mxx / lambda_), axis=0)

    return res


def value_function(working, period, x, endog_grid, value, beta, theta, duw):
    x = x.flatten("F")
    res = np.full(x.shape, np.nan)

    # Mark constrained region
    # credit constraint between 1st (M_{t+1) = 0) and second point (A_{t+1} = 0)
    mask = x < value[0]

    # Calculate t+1 value function in the constrained region
    # res[mask] = util(x[mask], working, theta, duw) + beta * value[0, 1]
    res[mask] = util(x[mask], working, theta, duw) + beta * endog_grid[1]

    # Calculate t+1 value function in non-constrained region
    # inter- and extrapolate
    interpolation = scin.interp1d(
        endog_grid,
        value,
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )
    res[~mask] = interpolation(x[~mask])

    return res


def util(consumption, working, theta, duw):
    """CRRA utility."""
    u = (consumption ** (1 - theta) - 1) / (1 - theta)
    u = u - duw * working

    return u


def income(it, shock, coeffs_age_poly):
    """Income in period t given normal shock."""

    ages = (it + 20) ** np.arange(len(coeffs_age_poly))
    w = np.exp(coeffs_age_poly @ ages + shock)
    return w


def simulate(
    endog_grid,
    policy,
    value,
    num_periods,
    # cost_work,
    # theta,
    # beta,
    lambda_,
    sigma,
    r,
    coeffs_age_poly,
    options,  # TO-DO
    params,  # TO-DO
    state_space,
    indexer,
    init=[10, 30],
    num_sims=10,
    seed=7134,
):
    # # To-Do. Fix for now, ignore exogenous proccess
    # endog_grid = endog_grid[::2]
    # policy = policy[::2]
    # value = value[::2]

    # partial
    discount_factor = params.loc[("beta", "beta"), "value"]
    delta = params.loc[("delta", "delta"), "value"]
    theta = params.loc[("utility_function", "theta"), "value"]
    compute_utility = partial(utility_func_crra, params=params)
    compute_value = partial(
        calc_current_value,
        discount_factor=discount_factor,
        compute_utility=compute_utility,
    )

    # Set seed
    np.random.seed(seed)

    # Create containers
    wealth0 = np.full((num_sims, num_periods), np.nan)
    wealth1 = np.full((num_sims, num_periods), np.nan)
    cons = np.full((num_sims, num_periods), np.nan)
    shock = np.full((num_sims, num_periods), np.nan)
    income_ = np.full((num_sims, num_periods), np.nan)
    worker = np.full((num_sims, num_periods), np.nan)
    prob_work = np.full((num_sims, num_periods), np.nan)
    ret_age = np.full((num_sims, 1), np.nan)
    vl1 = np.full((2, num_sims), np.nan)

    # Draw inperiodial wealth
    period = 0
    wealth0[:, 0] = init[0] + np.random.uniform(0, 1, num_sims) * (init[1] - init[0])

    # Set status of all individuals (given by nsims) to working, i.e. 1
    worker[:, 0] = 1

    # Fill in containers
    # Next period value function
    endog_grid_working = endog_grid[period, 0][~np.isnan(endog_grid[period, 0])]
    endog_grid_retired = endog_grid[period, 1][~np.isnan(endog_grid[period, 1])]
    value_working = value[period, 0][~np.isnan(value[period, 0])]
    value_retired = value[period, 1][~np.isnan(value[period, 1])]

    endog_grid_working_next = endog_grid[period + 1, 0][
        ~np.isnan(endog_grid[period + 1, 0])
    ]
    endog_grid_retired_next = endog_grid[period + 1, 1][
        ~np.isnan(endog_grid[period + 1, 1])
    ]
    policy_working_next = policy[period + 1, 0][~np.isnan(policy[period + 1, 0])]
    policy_retired_next = policy[period + 1, 1][~np.isnan(policy[period + 1, 1])]

    vl1[0] = value_function(
        0,
        period,
        wealth0[:, 0],
        endog_grid_retired,
        value_retired,
        discount_factor,
        theta,
        delta,
    )  # retirement
    vl1[1] = value_function(
        1,
        period,
        wealth0[:, 0],
        endog_grid_working,
        value_working,
        discount_factor,
        theta,
        delta,
    )  # work

    # Choice probabilperiody of working
    prob_work[:, 0] = choice_probabilities(vl1, lambda_)

    working = (prob_work[:, 0] > np.random.uniform(0, 1, num_sims)).astype(int)

    cons[:, 0][working == 0], cons[:, 0][working == 1] = cons_t0(
        wealth0,
        period,
        endog_grid_working_next,
        endog_grid_retired_next,
        policy_working_next,
        policy_retired_next,
        working,
    )
    wealth1[:, 0] = wealth0[:, 0] - cons[:, 0]

    # Record current period choice
    for period in range(1, num_periods - 1):
        # value_working = value[period, 0].T[~np.isnan(value[period, 0]).any(axis=0)].T
        # value_retired = value[period, 1].T[~np.isnan(value[period, 1]).any(axis=0)].T
        # policy_working_next = (
        #     policy[period + 1, 0].T[~np.isnan(policy[period + 1, 0]).any(axis=0)].T
        # )
        # policy_retired_next = (
        #     policy[period + 1, 1].T[~np.isnan(policy[period + 1, 1]).any(axis=0)].T
        # )

        # Fill in containers
        endog_grid_working = endog_grid[period, 0][~np.isnan(endog_grid[period, 0])]
        endog_grid_retired = endog_grid[period, 1][~np.isnan(endog_grid[period, 1])]
        value_working = value[period, 0][~np.isnan(value[period, 0])]
        value_retired = value[period, 1][~np.isnan(value[period, 1])]

        endog_grid_working_next = endog_grid[period + 1, 0][
            ~np.isnan(endog_grid[period + 1, 0])
        ]
        endog_grid_retired_next = endog_grid[period + 1, 1][
            ~np.isnan(endog_grid[period + 1, 1])
        ]
        policy_working_next = policy[period + 1, 0][~np.isnan(policy[period + 1, 0])]
        policy_retired_next = policy[period + 1, 1][~np.isnan(policy[period + 1, 1])]
        #

        worker[:, period] = working
        # Shock, here no shock since set sigma = 0 for m0
        shock[:, period][worker[:, period] == 1] = (
            norm.ppf(np.random.uniform(0, 1, sum(working))) * sigma
        )
        # Fill in retirement age
        ret_age[(worker[:, period - 1] == 1) & (worker[:, period] == 0)] = period

        wealth0, wealth1, cons, prob_work, working, shock, income_, vl1 = sim_periods(
            wealth0,
            wealth1,
            cons,
            worker,
            num_sims,
            prob_work,
            shock,
            period,
            coeffs_age_poly,
            lambda_,
            income_,
            endog_grid_working=endog_grid_working,
            endog_grid_retired=endog_grid_retired,
            value_working=value_working,
            value_retired=value_retired,
            endog_grid_working_next=endog_grid_working_next,
            endog_grid_retired_next=endog_grid_retired_next,
            policy_working_next=policy_working_next,
            policy_retired_next=policy_retired_next,
            # beta,
            # theta,
            # cost_work,
            r=r,
            # state_index=state_index,
            # state_index_plus_one=state_index_plus_one,
            # child_state=[period, working],  # TO-DO
            params=params,
            options=options,
            compute_value=compute_value,
        )
    df = create_dataframe(
        wealth0, wealth1, cons, worker, income_, shock, ret_age, num_sims, num_periods
    )

    return df


def sim_periods(
    wealth0,
    wealth1,
    cons,
    worker,
    num_sims,
    prob_work,
    shock,
    period,
    coeffs_age_poly,
    lambda_,
    income_,
    endog_grid_working,
    endog_grid_retired,
    value_working,
    value_retired,
    endog_grid_working_next,
    endog_grid_retired_next,
    policy_working_next,
    policy_retired_next,
    # beta,
    # theta,
    # cost_work,
    r,
    *,
    # child_state,
    # state_index,
    # state_index_plus_one,
    params,
    options,
    compute_value
):
    discount_factor = params.loc[("beta", "beta"), "value"]
    delta = params.loc[("delta", "delta"), "value"]
    theta = params.loc[("utility_function", "theta"), "value"]

    # Income
    # income_[:, period] = 0
    # income_[:, period][worker[:, period] == 1] = _calc_stochastic_income(
    #     # period, shock[:, period], coeffs_age_poly
    #     child_state=[period, 0],  # only worker
    #     wage_shock=shock[:, period],
    #     params=params,
    #     options=options,
    # )[worker[:, period] == 1]
    income_[:, period] = 0
    income_[:, period][worker[:, period] == 1] = income(
        period, shock[:, period], coeffs_age_poly
    )[worker[:, period] == 1]

    # M_t+1
    # MatLab code should be equvalent to calculating correct income for workers and retired
    # and just adding savings times interest
    # No extra need for further differentiating between retired and working
    wealth0[:, period] = income_[:, period] + wealth1[:, period - 1] * (1 + r)

    # Next period value function
    vl1 = np.full((2, num_sims), np.nan)

    # vl1[0, :] = value_function(
    #     0, period, wealth0[:, period], value_retired, discount_factor, theta, delta
    # )  # retirement
    # vl1[1, :] = value_function(
    #     1, period, wealth0[:, period], value_working, discount_factor, theta, delta
    # )  # work
    vl1[0] = value_function(
        0,
        period,
        wealth0[:, 0],
        endog_grid_retired,
        value_retired,
        discount_factor,
        theta,
        delta,
    )  # retirement
    vl1[1] = value_function(
        1,
        period,
        wealth0[:, 0],
        endog_grid_working,
        value_working,
        discount_factor,
        theta,
        delta,
    )  # work

    prob_work[:, period] = choice_probabilities(vl1, lambda_)

    working = (prob_work[:, period] > np.random.uniform(0, 1, num_sims)).astype(int)

    # retirement is absorbing state
    working[worker[:, period] == 0] = 0.0

    # # =================================================================================

    cons[:, period][working == 0], cons[:, period][working == 1] = cons_t0(
        wealth0,
        period,
        endog_grid_working_next,
        endog_grid_retired_next,
        policy_working_next,
        policy_retired_next,
        working,
    )

    # cons[:, period][working == 1] = cons11_flat
    # cons[:, period][working == 0] = cons10_flat

    wealth1[:, period] = wealth0[:, period] - cons[:, period]

    return wealth0, wealth1, cons, prob_work, working, shock, income_, vl1


def cons_t0(
    wealth0,
    period,
    endog_grid_working_next,
    endog_grid_retired_next,
    policy_working_next,
    policy_retired_next,
    working,
):
    """This function calculates the cons in period 0."""
    cons11 = np.interp(wealth0[:, period], endog_grid_working_next, policy_working_next)
    # breakpoint()
    # extrapolate linearly right of max grid point
    slope = (policy_working_next[-2] - policy_working_next[-1]) / (
        endog_grid_working_next[-2] - endog_grid_working_next[-1]
    )
    intercept = policy_working_next[-1] - endog_grid_working_next[-1] * slope
    cons11[cons11 == np.max(policy_working_next)] = (
        intercept + slope * wealth0[:, period][cons11 == np.max(policy_working_next)]
    )
    cons1_working_flat = cons11.flatten("F")

    # retirement
    cons10 = np.interp(wealth0[:, period], endog_grid_retired_next, policy_retired_next)
    # extrapolate linearly right of max grid point
    slope = (policy_retired_next[-2] - policy_retired_next[-1]) / (
        endog_grid_retired_next[-2] - endog_grid_retired_next[-1]
    )
    intercept = policy_retired_next[-1] - endog_grid_retired_next[-1] * slope
    cons10[cons10 == np.max(policy_retired_next)] = (
        intercept + slope * wealth0[:, period][cons10 == np.max(policy_retired_next)]
    )
    cons1_ret_flat = cons10.flatten("F")

    return cons1_ret_flat[working == 0], cons1_working_flat[working == 1]


# def cons_t(wealth0, period, policy_working_next, policy_retired_next, working):
#     """This function calculates the cons in period 0"""
#     cons11 = np.interp(
#         wealth0[:, period], policy_working_next[0], policy_working_next[1]
#     )
#     # breakpoint()
#     # extrapolate linearly right of max grid point
#     slope = (policy_working_next[1, -2] - policy_working_next[1, -1]) / (
#         policy_working_next[0, -2] - policy_working_next[0, -1]
#     )
#     intercept = policy_working_next[1, -1] - policy_working_next[0, -1] * slope
#     cons11[cons11 == np.max(policy_working_next[1])] = (
#         intercept + slope * wealth0[:, period][cons11 == np.max(policy_working_next[1])]
#     )
#     cons1_working_flat = cons11.flatten("F")

#     # retirement
#     cons10 = np.interp(
#         wealth0[:, period], policy_retired_next[0], policy_retired_next[1]
#     )
#     # extrapolate linearly right of max grid point
#     slope = (policy_retired_next[1, -2] - policy_retired_next[1, -1]) / (
#         policy_retired_next[0, -2] - policy_retired_next[0, -1]
#     )
#     intercept = policy_retired_next[1, -1] - policy_retired_next[0, -1] * slope
#     cons10[cons10 == np.max(policy_retired_next[1])] = (
#         intercept + slope * wealth0[:, period][cons10 == np.max(policy_retired_next[1])]
#     )
#     cons1_ret_flat = cons10.flatten("F")

#     return cons1_ret_flat[working == 0], cons1_working_flat[working == 1]


def create_dataframe(
    wealth0, wealth1, cons, worker, income_, shock, ret_age, num_sims, num_periods
):
    """This function processes the results so that they are composed in a pandas
    dataframe object."""

    # Set up multiindex object
    index = pd.MultiIndex.from_product(
        [np.arange(num_sims), np.arange(num_periods)], names=["identifier", "period"]
    )

    # Define column names object
    columns = [
        "wealth0",
        "wealth1",
        "consumption",
        "working",
        "income",
        "retirement_age",
        "shock",
    ]

    # Process data
    data = np.vstack(
        [
            wealth0.flatten("C"),
            wealth1.flatten("C"),
            cons.flatten("C"),
            worker.flatten("C"),
            income_.flatten("C"),
            ret_age.repeat(num_periods).flatten("C"),
            shock.flatten("C"),
        ]
    )

    df = pd.DataFrame(data.T, index, columns)

    return df
