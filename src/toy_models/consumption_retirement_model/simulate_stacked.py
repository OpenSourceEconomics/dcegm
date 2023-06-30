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
    # To-Do. Fix for now, ignore exogenous processes
    _endog_grid = endog_grid[::2]
    _value = value[::2]
    _policy = policy[::2]
    value = np.stack([_endog_grid, _value], axis=2)
    policy = np.stack([_endog_grid, _policy], axis=2)

    np.random.seed(seed)

    wealth_beginning_of_period = np.full((num_sims, num_periods), np.nan)
    wealth_next = np.full((num_sims, num_periods), np.nan)
    consumption = np.full((num_sims, num_periods), np.nan)
    wage_shock = np.full((num_sims, num_periods), np.nan)
    income_ = np.full((num_sims, num_periods), np.nan)
    worker = np.full((num_sims, num_periods), np.nan)
    prob_working = np.full((num_sims, num_periods), np.nan)
    retirement_age = np.full((num_sims, 1), np.nan)
    value_next = np.full((2, num_sims), np.nan)

    # Draw initial wealth
    period = 0
    wealth_beginning_of_period[:, period] = initial_wealth[0] + np.random.uniform(
        0, 1, num_sims
    ) * (initial_wealth[1] - initial_wealth[0])

    # Set status of all individuals to working, i.e. 1
    worker[:, 0] = 1

    value_working = value[period, 0].T[~np.isnan(value[period, 0]).any(axis=0)].T
    value_retired = value[period, 1].T[~np.isnan(value[period, 1]).any(axis=0)].T
    policy_working_next = (
        policy[period + 1, 0].T[~np.isnan(policy[period + 1, 0]).any(axis=0)].T
    )

    value_next[0, :] = get_next_value(
        choice=0,
        wealth=wealth_beginning_of_period[:, period],
        value_current=value_retired,
    )  # retirement
    value_next[1, :] = get_next_value(
        choice=1,
        wealth=wealth_beginning_of_period[:, period],
        value_current=value_working,
    )  # work

    prob_working[:, period] = calc_choice_probability(value_next, taste_shock_scale)

    labor_choice = (prob_working[:, period] > np.random.uniform(0, 1, num_sims)).astype(
        int
    )

    consumption[:, period][labor_choice == 1] = get_consumption(
        wealth_beginning_of_period[:, period], policy_working_next
    )[labor_choice == 1]

    wealth_next[:, period] = (
        wealth_beginning_of_period[:, period] - consumption[:, period]
    )

    for period in range(1, num_periods - 1):
        (
            wealth_beginning_of_period[:, period],
            wealth_next[:, period],
            consumption[:, period],
            prob_working[:, period],
            worker[:, period],
            labor_choice,
            wage_shock[:, period],
            income_[:, period],
            value_next,
            retirement_age,
        ) = simulate_period(
            policy=policy,
            value=value,
            wealth_from_previous_period=wealth_next[:, period - 1],
            consumption=consumption[:, period],
            retirement_age=retirement_age,
            worker=worker,
            choice_previous_period=labor_choice,
            num_sims=num_sims,
            prob_working=prob_working[:, period],
            wage_shock=wage_shock[:, period],
            period=period,
            coeffs_age_poly=coeffs_age_poly,
            lambda_=taste_shock_scale,
            sigma=sigma,
            labor_income=income_[:, period],
            r=r,
            get_next_value=get_next_value,
        )

    df = create_dataframe(
        wealth_beginning_of_period,
        wealth_next,
        consumption,
        worker,
        income_,
        wage_shock,
        retirement_age,
        num_sims,
        num_periods,
    )

    return df


def simulate_period(
    policy,
    value,
    wealth_from_previous_period,
    consumption,
    retirement_age,
    worker,
    choice_previous_period,
    num_sims,
    prob_working,
    wage_shock,
    period,
    *,
    coeffs_age_poly,
    lambda_,
    sigma,
    labor_income,
    r,
    get_next_value,
):
    """Simulate one period of the model."""

    value_working_current = (
        value[period, 0].T[~np.isnan(value[period, 0]).any(axis=0)].T
    )
    value_retired_current = (
        value[period, 1].T[~np.isnan(value[period, 1]).any(axis=0)].T
    )
    policy_working_next = (
        policy[period + 1, 0].T[~np.isnan(policy[period + 1, 0]).any(axis=0)].T
    )
    policy_retired_next = (
        policy[period + 1, 1].T[~np.isnan(policy[period + 1, 1]).any(axis=0)].T
    )

    worker_previous_period = worker[:, period - 1]
    worker = choice_previous_period

    wage_shock[worker == 1] = (
        norm.ppf(np.random.uniform(0, 1, sum(choice_previous_period))) * sigma
    )

    retirement_age[(worker_previous_period == 1) & (worker == 0)] = period

    #####
    labor_income[worker == 0] = 0
    labor_income[worker == 1] = calc_stochastic_income(
        period, wage_shock, coeffs_age_poly
    )[worker == 1]

    wealth_beginning_of_period = labor_income + wealth_from_previous_period * (1 + r)

    value_interp = np.full((2, num_sims), np.nan)
    value_interp[0, :] = get_next_value(
        choice=0,
        wealth=wealth_beginning_of_period,
        value_current=value_retired_current,
    )  # retirement
    value_interp[1, :] = get_next_value(
        choice=1,
        wealth=wealth_beginning_of_period,
        value_current=value_working_current,
    )  # work

    prob_working = calc_choice_probability(value_interp, lambda_)

    choice_previous_period = (prob_working > np.random.uniform(0, 1, num_sims)).astype(
        int
    )

    # retirement is absorbing state
    choice_previous_period[worker == 0] = 0

    consumption[choice_previous_period == 1] = get_consumption(
        wealth_beginning_of_period, policy_working_next
    )[choice_previous_period == 1]
    consumption[choice_previous_period == 0] = get_consumption(
        wealth_beginning_of_period, policy_retired_next
    )[choice_previous_period == 0]

    wealth_end_of_period = wealth_beginning_of_period - consumption

    return (
        wealth_beginning_of_period,
        wealth_end_of_period,
        consumption,
        prob_working,
        worker,
        choice_previous_period,
        wage_shock,
        labor_income,
        value_interp,
        retirement_age,
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
