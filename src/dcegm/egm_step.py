"""Implementation of the EGM algorithm."""
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.consumption_retirement_model import calc_next_period_marginal_wealth
from dcegm.consumption_retirement_model import get_next_period_wealth_matrices
from scipy import interpolate


def do_egm_step(
    choice: int,
    state: np.ndarray,
    *,
    params: pd.DataFrame,
    options: Dict[str, int],
    exogenous_grid: Dict[str, np.ndarray],
    utility_functions: Dict[str, callable],
    compute_expected_value: Callable,
    next_period_policy: np.ndarray,
    next_period_value: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the Endogenous-Grid-Method Algorithm (EGM step).

    Args:
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        state (np.ndarray): Current individual state.
        params (pd.DataFrame): Model parameters indexed with multi-index of the form
            ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        exogenous_grid (Dict[str, np.ndarray]): Dictionary containing the
            exogenous grids of (i) savings (array of shape (n_grid_wealth, ))
            (ii) quadrature points (array of shape (n_quad_stochastic, )) and
            (iii) associated quadrature weights (also an array of shape
            (n_quad_stochastic, )).
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of (i) utility, (ii) inverse marginal utility,
            and (iii) next period marginal utility.
        compute_expected_value (callable): User-supplied functions for computation
            of the agent's expected value.
        next_period_policy (np.ndarray): Array of the next period policy
            for all choices. Shape (n_choices, 2, 1.1 * (n_grid_wealth + 1)).
        next_period_value (np.ndarray): Array of the next period values
            for all choices. Shape (n_choices, 2, 1.1 * (n_grid_wealth + 1)).

    Returns:
        (tuple) Tuple containing

        - policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*].
        - value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*].
        - expected_value (np.ndarray): The expected value of continuation.

    """
    n_grid_wealth = options["grid_points_wealth"]
    current_policy = np.empty((2, n_grid_wealth + 1))
    current_value = np.empty((2, n_grid_wealth + 1))

    # 0) Preliminaries
    # Matrices of all possible next period wealths and marginal wealths
    matrix_next_period_wealth = get_next_period_wealth_matrices(
        state,
        choice,
        params=params,
        options=options,
        savings=exogenous_grid["savings"],
        quad_points=exogenous_grid["quadrature_points"],
    )

    next_period_marginal_wealth = calc_next_period_marginal_wealth(
        state, params, options
    )

    # Interpolate next period policy and values to match the
    # contemporary matrix of potential next period wealths
    next_period_policy = get_next_period_policy(
        matrix_next_period_wealth,
        next_period_policy=next_period_policy,
        options=options,
    )

    next_period_value = get_next_period_value(
        matrix_next_period_wealth=matrix_next_period_wealth,
        period=state[0],
        params=params,
        options=options,
        next_period_value=next_period_value,
        compute_utility=utility_functions["utility"],
    )

    # i) Current period consumption & endogenous wealth grid
    current_period_policy = get_current_period_policy(
        choice,
        next_period_policy=next_period_policy,
        matrix_next_period_wealth=matrix_next_period_wealth,
        next_period_marginal_wealth=next_period_marginal_wealth,
        next_period_value=next_period_value,
        params=params,
        options=options,
        quad_weights=exogenous_grid["quadrature_weights"],
        utility_functions=utility_functions,
    )
    endog_wealth_grid = get_endogenous_wealth_grid(
        current_period_policy, exog_savings_grid=exogenous_grid["savings"]
    )

    # ii) Expected & current period value
    expected_value, current_period_value = get_expected_and_current_period_value(
        choice,
        next_period_value=next_period_value,
        matrix_next_period_wealth=matrix_next_period_wealth,
        current_period_policy=current_period_policy,
        quad_weights=exogenous_grid["quadrature_weights"],
        params=params,
        options=options,
        compute_utility=utility_functions["utility"],
        compute_expected_value=compute_expected_value,
    )

    current_policy[0, 1:] = endog_wealth_grid
    current_policy[1, 1:] = current_period_policy

    current_value[0, 1:] = endog_wealth_grid
    current_value[1, 1:] = current_period_value

    current_policy, current_value = _set_first_elements(
        current_policy, current_value, expected_value
    )

    return current_policy, current_value, expected_value


def get_next_period_policy(
    matrix_next_period_wealth: np.ndarray,
    next_period_policy: np.ndarray,
    options: Dict[str, int],
) -> np.ndarray:
    """Computes the next-period policy via linear interpolation.

    Extrapolate lineary in wealth regions beyond the grid, i.e. larger
    than "max_wealth" specifiec in the ``params`` dictionary.

    Args:
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_policy (np.ndarray): Array of the next period policy
            for all choices. Shape (n_choices, 2, 1.1 * n_grid_wealth + 1).
        options (dict): Options dictionary.

    Returns:
        next_period_policy_interp (np.ndarray): Array of interpolated next period
            consumption of shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """
    n_choices = options["n_discrete_choices"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]

    next_period_policy_interp = np.empty((n_choices, n_quad_stochastic * n_grid_wealth))

    for index in range(n_choices):
        next_period_policy_interp[index, :] = interpolate_policy(
            matrix_next_period_wealth.flatten("F"), next_period_policy[index]
        )

    return next_period_policy_interp


def get_next_period_value(
    matrix_next_period_wealth: np.ndarray,
    next_period_value: np.ndarray,
    period: int,
    params: pd.DataFrame,
    options: Dict[str, int],
    compute_utility: Callable,
) -> np.ndarray:
    """Maps next-period value onto this period's matrix of next-period wealth.

    Args:
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_value (np.ndarray): Array of the next period value
            for all choices. Shape (n_choices, 2, 1.1 * n_grid_wealth + 1).
        period (int): Current period t.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        compute_utility (callable): User-supplied functions for computation
            of the agent's utility.

    Returns:
        next_period_value_interp (np.ndarray): Array containing interpolated
            values of next period choice-specific value function. We use
            interpolation to the actual next period value function onto
            the current period grid of potential next period wealths.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """
    n_choices = options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else range(n_choices)

    next_period_value_interp = np.empty(
        (
            n_choices,
            matrix_next_period_wealth.shape[0] * matrix_next_period_wealth.shape[1],
        )
    )

    for index, choice in enumerate(choice_range):
        if period == options["n_periods"] - 2:
            next_period_value_interp[index, :] = compute_utility(
                matrix_next_period_wealth.flatten("F"), choice, params
            )
        else:
            next_period_value_interp[index, :] = interpolate_value(
                flat_wealth=matrix_next_period_wealth.flatten("F"),
                value=next_period_value[index],
                choice=choice,
                params=params,
                compute_utility=compute_utility,
            )

    return next_period_value_interp


def get_current_period_policy(
    choice: int,
    next_period_policy: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    next_period_marginal_wealth: np.ndarray,
    next_period_value: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    quad_weights: np.ndarray,
    utility_functions: Dict[str, callable],
) -> np.ndarray:
    """Computes the current period policy.

    Args:
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        true_next_period_policy (List[np.ndarray]): Nested list of np.ndarrays storing
            choice-specific consumption policies. Dimensions of the list are:
            [n_discrete_choices][2, *n_endog_wealth_grid*], where
            *n_endog_wealth_grid* is of variable length depending on the number of
            concurrent local optima for consumption. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which we set
            to zero (that's why we have n_grid_wealth + 1 initial points).
            Position [0, :] of the arrays contain the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the (consumption) policy
            function c(M, d), for each time period and each discrete choice.
            exogenous savings grid.
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_marginal_wealth (np.ndarray): Array of all possible next period
            marginal wealths. Also of shape (n_quad_stochastic, n_grid_wealth)
        next_period_value (np.ndarray): Array containing interpolated
            values of next period choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        quad_weights (np.ndarray): Weights associated with the quadrature points
            of shape (n_quad_stochastic,). Used for integration over the
            stochastic income component in the Euler equation.
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of (i) utility, (ii) inverse marginal utility,
            and (iii) next period marginal utility.

    Returns:
        current_period_policy (np.ndarray): Policy (consumption) in the current
            period. Array of shape (n_grid_wealth,).
    """
    beta = params.loc[("beta", "beta"), "value"]
    _inv_marg_utility_func = utility_functions["inverse_marginal_utility"]
    _compute_next_period_marginal_utility = utility_functions[
        "next_period_marginal_utility"
    ]

    next_period_marginal_utility = _compute_next_period_marginal_utility(
        choice,
        next_period_consumption=next_period_policy,
        next_period_value=next_period_value,
        params=params,
        options=options,
    )

    # RHS of Euler Eq., p. 337 IJRS (2017)
    # Integrate out uncertainty over stochastic income y
    rhs_euler = _calc_rhs_euler(
        next_period_marginal_utility,
        matrix_next_period_wealth=matrix_next_period_wealth,
        next_period_marginal_wealth=next_period_marginal_wealth,
        quad_weights=quad_weights,
    )
    current_period_policy = _inv_marg_utility_func(
        marginal_utility=beta * rhs_euler, params=params
    )

    return current_period_policy


def get_endogenous_wealth_grid(
    current_period_policy: np.ndarray, exog_savings_grid: np.ndarray
) -> np.ndarray:
    """Returns the endogenous wealth grid of the current period.
    .
    Args:
        current_period_policy (np.ndarray): Consumption in the current
            period. Array of shape (n_grid_wealth,).
        exog_savings_grid (np.ndarray): Exogenous grid over savings.
            Array of shape (n_grid_wealth,).

    Returns:
        endog_wealth_grid (np.ndarray): Endogenous wealth grid of shape
            (n_grid_wealth,).
    """
    endog_wealth_grid = exog_savings_grid + current_period_policy

    return endog_wealth_grid


def get_expected_and_current_period_value(
    choice: int,
    next_period_value: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    current_period_policy: np.ndarray,
    quad_weights: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    compute_utility: Callable,
    compute_expected_value: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the expected (next-period) value and the current period's value.

    Args:
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        current_period_policy (np.ndarray): Consumption in the current
            period. Array of shape (n_grid_wealth,).
        quad_weights (np.ndarray): Weights associated with the quadrature points
            of shape (n_quad_stochastic,). Used for integration over the
            stochastic income component in the Euler equation.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        value_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of the agent's (i) value function before
            the final period, (ii) value function in the final period,
            and (iii) the expected value.

    Returns:
        expected_value (np.ndarray): Expected value of next period. Array of
            shape (n_grid_wealth,).
        current_period_value (np.ndarray): Array of current period value
            function of shape (n_grid_wealth,).
    """
    beta = params.loc[("beta", "beta"), "value"]

    expected_value = compute_expected_value(
        choice,
        matrix_next_period_wealth,
        next_period_value=next_period_value,
        quad_weights=quad_weights,
        params=params,
        options=options,
    )

    utility = compute_utility(current_period_policy, choice, params)
    current_period_value = utility + beta * expected_value

    return expected_value, current_period_value


def interpolate_policy(flat_wealth: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """Interpolate the agent's policy for given flat wealth matrix.

    Args:
        flat_wealth (np.ndarray): Flat array of shape
            (n_quad_stochastic *n_grid_wealth,) containing the agent's
            potential wealth matrix in given period.
        policy (np.ndarray): Policy array of shape (2, 1.1 * n_grid_wealth).
            Position [0, :] of the arrays contain the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the (consumption) policy
            function c(M, d), for each time period and each discrete choice.
    """
    policy = policy[:, ~np.isnan(policy).any(axis=0)]

    interpolation_func = interpolate.interp1d(
        x=policy[0, :],
        y=policy[1, :],
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )

    policy_interp = interpolation_func(flat_wealth)

    return policy_interp


def interpolate_value(
    flat_wealth: np.ndarray,
    value: np.ndarray,
    choice: int,
    params: pd.DataFrame,
    compute_utility: Callable,
) -> np.ndarray:
    """Interpolate the agent's value for given flat wealth matrix.

    Args:
        flat_wealth (np.ndarray): Flat array of shape
            (n_quad_stochastic *n_grid_wealth,) containing the agent's
            potential wealth matrix in given period.
        value (np.ndarray): Value array of shape (2, 1.1* n_grid_wealth).
            Position [0, :] of the array contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the value function v(M, d),
            for each time period and each discrete choice.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        compute_utility (callable): Function for computation of agent's utility.
    """
    value = value[:, ~np.isnan(value).any(axis=0)]
    value_interp = np.empty(flat_wealth.shape)

    # Mark credit constrained region
    constrained_region = flat_wealth < value[0, 1]

    # Calculate t+1 value function in constrained region using
    # the analytical part
    value_interp[constrained_region] = _get_value_constrained(
        flat_wealth[constrained_region],
        next_period_value=value[1, 0],
        choice=choice,
        params=params,
        compute_utility=compute_utility,
    )

    # Calculate t+1 value function in non-constrained region
    # via inter- and extrapolation
    interpolation_func = interpolate.interp1d(
        x=value[0, :],  # endogenous wealth grid
        y=value[1, :],  # value_function
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )
    value_interp[~constrained_region] = interpolation_func(
        flat_wealth[~constrained_region]
    )

    return value_interp


def _get_value_constrained(
    wealth: np.ndarray,
    next_period_value: np.ndarray,
    choice: int,
    params: pd.DataFrame,
    compute_utility: Callable,
) -> np.ndarray:
    """Compute the agent's value in the credit constrained region."""
    beta = params.loc[("beta", "beta"), "value"]

    utility = compute_utility(wealth, choice, params)
    value_constrained = utility + beta * next_period_value

    return value_constrained


def _calc_rhs_euler(
    next_period_marginal_utility: np.ndarray,
    matrix_next_period_wealth: np.ndarray,
    next_period_marginal_wealth: np.ndarray,
    quad_weights: np.ndarray,
) -> np.ndarray:
    """Computes the right-hand side of the Euler equation, p. 337 IJRS (2017).

    Args:
        next_period_marginal_utility (np.ndarray): Array of next period's
            marginal utility of shape (n_quad_stochastic * n_grid_wealth,).
        matrix_next_period_wealth(np.ndarray): Array of all possible next
            period wealths. Shape (n_quad_stochastic, n_wealth_grid).
        next_period_marginal_wealth(np.ndarray): Array of marginal next period wealths.
            Shape (n_quad_stochastic, n_wealth_grid).
        quad_weights (np.ndarray): Weights associated with the quadrature points
            of shape (n_quad_stochastic,). Used for integration over the
            stochastic income component in the Euler equation.

    Returns:
        rhs_euler (np.ndarray): Right-hand side of the Euler equation.
            Shape (n_grid_wealth,).
    """
    next_period_marginal_utility = next_period_marginal_utility.reshape(
        matrix_next_period_wealth.shape, order="F"
    )

    rhs_euler = np.dot(
        quad_weights.T,
        np.multiply(next_period_marginal_utility, next_period_marginal_wealth),
    )

    return rhs_euler


def _set_first_elements(
    policy: np.ndarray, value: np.ndarray, expected_value: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Sets first value to expected value and the other elements to zero."""
    policy[0, 0] = 0
    policy[1, 0] = 0

    value[0, 0] = 0
    value[1, 0] = expected_value[0]

    return policy, value
