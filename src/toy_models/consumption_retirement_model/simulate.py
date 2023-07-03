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
    coeffs_age_poly,
    wage_shock_scale,
    taste_shock_scale,
    initial_wealth,
    interest_rate,
    n_periods,
    state_space,
    indexer,
    interpolate_value_on_current_grid,
    n_sims=10,
    seed=7134,
):
    """Simulate the model."""
    np.random.seed(seed)

    # To-Do. For now, ignore exogenous processes
    _endog_grid = endog_grid[::2]
    _value = value[::2]
    _policy = policy[::2]
    value = np.stack([_endog_grid, _value], axis=2)
    policy = np.stack([_endog_grid, _policy], axis=2)

    wealth_beginning_of_period = np.full((n_sims, n_periods), np.nan)
    wealth_end_of_period = np.full((n_sims, n_periods), np.nan)
    consumption = np.full((n_sims, n_periods), np.nan)
    wage_shock = np.full((n_sims, n_periods), np.nan)
    labor_income = np.full((n_sims, n_periods), np.nan)
    lagged_choices = np.full((n_sims, n_periods), np.nan)
    prob_working = np.full((n_sims, n_periods), np.nan)
    retirement_age = np.full((n_sims, 1), np.nan)

    # Draw initial wealth
    period = 0
    wealth_beginning_of_period[:, period] = initial_wealth[0] + np.random.uniform(
        0, 1, n_sims
    ) * (initial_wealth[1] - initial_wealth[0])

    # Set status of all individuals to working, i.e. 1
    lagged_choices[:, 0] = 1

    value_working = value[period, 0].T[~np.isnan(value[period, 0]).any(axis=0)].T
    value_retired = value[period, 1].T[~np.isnan(value[period, 1]).any(axis=0)].T
    policy_working_next = (
        policy[period + 1, 0].T[~np.isnan(policy[period + 1, 0]).any(axis=0)].T
    )

    _values_interp_stacked = np.full((2, n_sims), np.nan)
    _values_interp_stacked[0, :] = interpolate_value_on_current_grid(
        choice=0,
        wealth=wealth_beginning_of_period[:, period],
        value_current=value_retired,
    )  # retirement
    _values_interp_stacked[1, :] = interpolate_value_on_current_grid(
        choice=1,
        wealth=wealth_beginning_of_period[:, period],
        value_current=value_working,
    )  # work

    prob_working[:, period] = calc_choice_probability(
        _values_interp_stacked, taste_shock_scale
    )
    labor_choice = (prob_working[:, period] > np.random.uniform(0, 1, n_sims)).astype(
        int
    )

    consumption[:, period][labor_choice == 1] = get_consumption(
        wealth_beginning_of_period[:, period], policy_working_next
    )[
        labor_choice == 1
    ]  # work

    wealth_end_of_period[:, period] = (
        wealth_beginning_of_period[:, period] - consumption[:, period]
    )

    for period in range(1, n_periods - 1):
        (
            wealth_beginning_of_period[:, period],
            wealth_end_of_period[:, period],
            consumption[:, period],
            prob_working[:, period],
            lagged_choices[:, period],
            labor_choice,
            wage_shock[:, period],
            labor_income[:, period],
            retirement_age,
        ) = simulate_period(
            policy=policy,
            value=value,
            wealth_from_previous_period=wealth_end_of_period[:, period - 1],
            consumption=consumption[:, period],
            retirement_age=retirement_age,
            choice_two_periods_ago=lagged_choices[:, period - 1],
            lagged_choice=labor_choice,
            num_sims=n_sims,
            prob_working=prob_working[:, period],
            wage_shock=wage_shock[:, period],
            period=period,
            coeffs_age_poly=coeffs_age_poly,
            taste_shock_scale=taste_shock_scale,
            wage_shock_scale=wage_shock_scale,
            labor_income=labor_income[:, period],
            interest_rate=interest_rate,
            interpolate_value_on_current_grid=interpolate_value_on_current_grid,
        )

    df = create_dataframe(
        wealth_beginning_of_period,
        wealth_end_of_period,
        consumption,
        lagged_choices,
        labor_income,
        wage_shock,
        retirement_age,
        n_sims,
        n_periods,
    )

    return df


def simulate_period(
    policy,
    value,
    wealth_from_previous_period,
    consumption,
    retirement_age,
    choice_two_periods_ago,
    lagged_choice,
    num_sims,
    prob_working,
    wage_shock,
    period,
    coeffs_age_poly,
    taste_shock_scale,
    wage_shock_scale,
    labor_income,
    interest_rate,
    interpolate_value_on_current_grid,
):
    """Simulate one period of the model."""

    _value_current_working = (
        value[period, 0].T[~np.isnan(value[period, 0]).any(axis=0)].T
    )
    _value_current_retired = (
        value[period, 1].T[~np.isnan(value[period, 1]).any(axis=0)].T
    )
    _policy_next_working = (
        policy[period + 1, 0].T[~np.isnan(policy[period + 1, 0]).any(axis=0)].T
    )
    _policy_next_retired = (
        policy[period + 1, 1].T[~np.isnan(policy[period + 1, 1]).any(axis=0)].T
    )

    retirement_age[(choice_two_periods_ago == 1) & (lagged_choice == 0)] = period

    wage_shock[lagged_choice == 1] = (
        norm.ppf(np.random.uniform(0, 1, sum(lagged_choice))) * wage_shock_scale
    )
    labor_income[lagged_choice == 0] = 0
    labor_income[lagged_choice == 1] = calc_stochastic_income(
        period, wage_shock, coeffs_age_poly
    )[lagged_choice == 1]

    wealth_beginning_of_period = labor_income + wealth_from_previous_period * (
        1 + interest_rate
    )

    _values_interp_stacked = np.full((2, num_sims), np.nan)
    _values_interp_stacked[0, :] = interpolate_value_on_current_grid(
        choice=0,
        wealth=wealth_beginning_of_period,
        value_current=_value_current_retired,
    )  # retirement
    _values_interp_stacked[1, :] = interpolate_value_on_current_grid(
        choice=1,
        wealth=wealth_beginning_of_period,
        value_current=_value_current_working,
    )  # work

    prob_working = calc_choice_probability(_values_interp_stacked, taste_shock_scale)
    choice_current_period = (prob_working > np.random.uniform(0, 1, num_sims)).astype(
        int
    )
    choice_current_period[lagged_choice == 0] = 0  # retirement is absorbing state

    consumption[choice_current_period == 0] = get_consumption(
        wealth_beginning_of_period, _policy_next_retired
    )[
        choice_current_period == 0
    ]  # retirement
    consumption[choice_current_period == 1] = get_consumption(
        wealth_beginning_of_period, _policy_next_working
    )[
        choice_current_period == 1
    ]  # work

    wealth_end_of_period = wealth_beginning_of_period - consumption

    return (
        wealth_beginning_of_period,
        wealth_end_of_period,
        consumption,
        prob_working,
        lagged_choice,
        choice_current_period,
        wage_shock,
        labor_income,
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
    lagged_labor_choice,
    labor_income,
    wage_shock,
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
        "lagged_labor_choice",
        "labor_income",
        "retirement_age",
        "wage_shock",
    ]

    data_raw = np.vstack(
        [
            wealth_current.flatten("C"),
            wealth_next.flatten("C"),
            consumption.flatten("C"),
            lagged_labor_choice.flatten("C"),
            labor_income.flatten("C"),
            retirement_age.repeat(n_periods).flatten("C"),
            wage_shock.flatten("C"),
        ]
    )

    return pd.DataFrame(data_raw.T, index, columns)
