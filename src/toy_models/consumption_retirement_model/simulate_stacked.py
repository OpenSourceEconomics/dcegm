"""Simulation function for the consumption and retirement toy model."""
import numpy as np
import pandas as pd
from dcegm.interpolation import linear_interpolation_with_extrapolation
from scipy.stats import norm


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


# def interpolate_value(
#     value_high: float,
#     wealth_high: float,
#     value_low: float,
#     wealth_low: float,
#     wealth_new: float,
# ):
#     """Interpolate policy and value functions.

#     Args:
#         policy_high (float): Policy function value at the higher end of the
#             interpolation interval.
#         value_high (float): Value function value at the higher end of the
#             interpolation interval.
#         wealth_high (float): Wealth value at the higher end of the interpolation
#             interval.
#         policy_low (float): Policy function value at the lower end of the
#             interpolation interval.
#         value_low (float): Value function value at the lower end of the
#             interpolation interval.
#         wealth_low (float): Wealth value at the lower end of the interpolation
#             interval.
#         wealth_new (float): Wealth value at which the policy and value functions
#             should be interpolated.

#     Returns:
#         tuple:

#         - policy_new (float): Interpolated policy function value.
#         - value_new (float): Interpolated value function value.

#     """
#     interpolate_dist = wealth_new - wealth_low
#     interpolate_slope_value = (value_high - value_low) / (wealth_high - wealth_low)
#     value_new = (interpolate_slope_value * interpolate_dist) + value_low

#     return value_new


def calc_stochastic_income(period: int, wage_shock: float, coeffs_age_poly: np.ndarray):
    """Computes the current level of stochastic labor income.

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
        params_dict (dict): Dictionary containing model parameters.
            Relevant here are the coefficients of the wage equation.
        options (dict): Options dictionary.

    Returns:
        stochastic_income (float): The potential end of period income. It consists of a
            deterministic component, i.e. age-dependent labor income,
            and a stochastic shock.

    """
    age = period + 20
    labor_income = coeffs_age_poly @ (age ** np.arange(len(coeffs_age_poly)))
    return np.exp(labor_income + wage_shock)


def simulate_stacked(
    endog_grid,
    value,
    policy,
    num_periods,
    taste_shock_scale,
    sigma,
    r,
    coeffs_age_poly,
    initial_wealth,
    state_space,
    indexer,
    get_next_value,
    num_sims=10,
    seed=7134,
):
    """Simulate the model."""
    # To-Do. Fix for now, ignore exogenous proccessoen
    _endog_grid = endog_grid[::2]
    _value = value[::2]
    _policy = policy[::2]
    value = np.stack([_endog_grid, _value], axis=2)
    policy = np.stack([_endog_grid, _policy], axis=2)

    np.random.seed(seed)

    wealth_current = np.full((num_sims, num_periods), np.nan)
    wealth_next = np.full((num_sims, num_periods), np.nan)
    consumption = np.full((num_sims, num_periods), np.nan)
    income_shock = np.full((num_sims, num_periods), np.nan)
    income_ = np.full((num_sims, num_periods), np.nan)
    worker = np.full((num_sims, num_periods), np.nan)
    prob_working = np.full((num_sims, num_periods), np.nan)
    retirement_age = np.full((num_sims, 1), np.nan)
    value_next = np.full((2, num_sims), np.nan)

    # Draw initial wealth
    period = 0
    wealth_current[:, period] = initial_wealth[0] + np.random.uniform(
        0, 1, num_sims
    ) * (initial_wealth[1] - initial_wealth[0])

    # Set status of all individuals to working, i.e. 1
    worker[:, 0] = 1

    value_working = value[period, 0].T[~np.isnan(value[period, 0]).any(axis=0)].T
    value_retired = value[period, 1].T[~np.isnan(value[period, 1]).any(axis=0)].T
    policy_working_next = (
        policy[period + 1, 0].T[~np.isnan(policy[period + 1, 0]).any(axis=0)].T
    )
    policy_retired_next = (
        policy[period + 1, 1].T[~np.isnan(policy[period + 1, 1]).any(axis=0)].T
    )

    value_next[0, :] = get_next_value(
        choice=0,
        wealth=wealth_current[:, period],
        value_current=value_retired,
    )  # retirement
    value_next[1, :] = get_next_value(
        choice=1,
        wealth=wealth_current[:, period],
        value_current=value_working,
    )  # work

    prob_working[:, period] = calc_choice_probability(value_next, taste_shock_scale)

    working = (prob_working[:, period] > np.random.uniform(0, 1, num_sims)).astype(int)

    consumption[:, period][working == 1] = get_consumption(
        wealth_current[:, period], policy_working_next
    )[working == 1]

    wealth_next[:, period] = wealth_current[:, period] - consumption[:, period]

    for period in range(1, num_periods - 1):
        value_working = value[period, 0].T[~np.isnan(value[period, 0]).any(axis=0)].T
        value_retired = value[period, 1].T[~np.isnan(value[period, 1]).any(axis=0)].T
        policy_working_next = (
            policy[period + 1, 0].T[~np.isnan(policy[period + 1, 0]).any(axis=0)].T
        )
        policy_retired_next = (
            policy[period + 1, 1].T[~np.isnan(policy[period + 1, 1]).any(axis=0)].T
        )

        worker[:, period] = working

        income_shock[:, period][worker[:, period] == 1] = (
            norm.ppf(np.random.uniform(0, 1, sum(working))) * sigma
        )

        retirement_age[(worker[:, period - 1] == 1) & (worker[:, period] == 0)] = period

        (
            wealth_current,
            wealth_next,
            consumption,
            prob_working,
            working,
            income_shock,
            income_,
            value_next,
        ) = simulate_period(
            wealth_current,
            wealth_next,
            consumption,
            worker,
            num_sims,
            prob_working,
            income_shock,
            period,
            coeffs_age_poly=coeffs_age_poly,
            lambda_=taste_shock_scale,
            income_=income_,
            value_working=value_working,
            value_retired=value_retired,
            policy_working_next=policy_working_next,
            policy_retired_next=policy_retired_next,
            r=r,
            get_next_value=get_next_value,
        )
    df = create_dataframe(
        wealth_current,
        wealth_next,
        consumption,
        worker,
        income_,
        income_shock,
        retirement_age,
        num_sims,
        num_periods,
    )

    return df


def simulate_period(
    wealth_current,
    wealth_next,
    consumption,
    worker,
    num_sims,
    prob_working,
    income_shock,
    period,
    *,
    coeffs_age_poly,
    lambda_,
    income_,
    value_working,
    value_retired,
    policy_working_next,
    policy_retired_next,
    r,
    get_next_value,
):
    """Simulate one period of the model."""
    income_[:, period] = 0
    income_[:, period][worker[:, period] == 1] = calc_stochastic_income(
        period, income_shock[:, period], coeffs_age_poly
    )[worker[:, period] == 1]

    wealth_current[:, period] = income_[:, period] + wealth_next[:, period - 1] * (
        1 + r
    )

    value_next = np.full((2, num_sims), np.nan)
    value_next[0, :] = get_next_value(
        choice=0,
        wealth=wealth_current[:, period],
        value_current=value_retired,
    )  # retirement
    value_next[1, :] = get_next_value(
        choice=1,
        wealth=wealth_current[:, period],
        value_current=value_working,
    )  # work

    prob_working[:, period] = calc_choice_probability(value_next, lambda_)

    working = (prob_working[:, period] > np.random.uniform(0, 1, num_sims)).astype(int)

    # retirement is absorbing state
    working[worker[:, period] == 0] = 0.0

    consumption[:, period][working == 1] = get_consumption(
        wealth_current[:, period], policy_working_next
    )[working == 1]
    consumption[:, period][working == 0] = get_consumption(
        wealth_current[:, period], policy_retired_next
    )[working == 0]

    wealth_next[:, period] = wealth_current[:, period] - consumption[:, period]

    return (
        wealth_current,
        wealth_next,
        consumption,
        prob_working,
        working,
        income_shock,
        income_,
        value_next,
    )


def get_consumption(wealth_current, policy_next):
    """Calculate consumption in period t."""
    consumption = linear_interpolation_with_extrapolation(
        x=policy_next[0], y=policy_next[1], x_new=wealth_current
    )

    if np.any(consumption == np.max(policy_next[1])):
        slope = (policy_next[1, -2] - policy_next[1, -1]) / (
            policy_next[0, -2] - policy_next[0, -1]
        )
        intercept = policy_next[1, -1] - policy_next[0, -1] * slope
        consumption[consumption == np.max(policy_next[1])] = (
            intercept + slope * wealth_current[consumption == np.max(policy_next[1])]
        )

    return consumption


def create_dataframe(
    wealth_current,
    wealth_next,
    consumption,
    worker,
    labor_income,
    shock,
    retirement_age,
    n_sims,
    n_periods,
):
    """Combine the simulate results in a pandas DataFrame."""

    index = pd.MultiIndex.from_product(
        [np.arange(n_sims), np.arange(n_periods)], names=["person_identifier", "period"]
    )

    columns = [
        "wealth_beginning_of_period",
        "wealth_end_of_period",
        "consumption",
        "working",
        "income",
        "retirement_age",
        "shock",
    ]

    data_raw = np.vstack(
        [
            wealth_current.flatten("C"),
            wealth_next.flatten("C"),
            consumption.flatten("C"),
            worker.flatten("C"),
            labor_income.flatten("C"),
            retirement_age.repeat(n_periods).flatten("C"),
            shock.flatten("C"),
        ]
    )

    return pd.DataFrame(data_raw.T, index, columns)
