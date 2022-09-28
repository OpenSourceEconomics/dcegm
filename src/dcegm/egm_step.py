"""Implementation of the EGM algorithm."""
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.consumption_retirement_model import calc_next_period_marginal_wealth
from dcegm.consumption_retirement_model import get_next_period_wealth_matrices


def do_egm_step(
    choice: int,
    state: np.ndarray,
    *,
    params: pd.DataFrame,
    options: Dict[str, int],
    exogenous_grid: Dict[str, np.ndarray],
    utility_functions: Dict[str, callable],
    compute_expected_value: Callable,
    next_period_policy_function: Dict[int, Callable],
    next_period_value_function: Dict[int, Callable]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Runs the Endogenous-Grid-Method Algorithm (EGM step).
    Args:
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
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
        next_period_policy_function (Dict[int, callable]): Dictionary of partial
            functions (one for each discrete choice) functions to compute agent's
            choice-specific policy function of the next period.
            It takes the current period matrix of potential next-period wealths
            as input.
        next_period_value_function (Dict[int, callable]): Dictionary of partial
            functions (one for each discrete choice) functions to compute agent's
            choice-specific value function of the next period.
            It takes the current period matrix of potential next-period wealths
            as input.

    Returns:
        (tuple) Tuple containing

        - policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*].
        - value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*].
        - expected_value: The expected value of continuation

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
    next_period_value = get_next_period_value(
        matrix_next_period_wealth=matrix_next_period_wealth,
        options=options,
        next_period_value_function=next_period_value_function,
    )

    next_period_policy = get_next_period_policy(
        matrix_next_period_wealth,
        options=options,
        next_period_policy_function=next_period_policy_function,
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


def get_next_period_value(
    matrix_next_period_wealth: np.ndarray,
    options: Dict[str, int],
    next_period_value_function: Dict[int, callable],
) -> np.ndarray:
    """Maps next-period value onto this period's matrix of next-period wealth.

    Args:
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        options (dict): Options dictionary.
        next_period_value_function (Dict[int, callable]): Dictionary of partial
            functions (one for each discrete choice) functions to compute agent's
            choice-specific value function of the next period.
            It takes the current period matrix of potential next-period wealths
            as input.

    Returns:
        next_period_value_interp (np.ndarray): Array containing interpolated
            values of next period choice-specific value function. We use
            interpolation to the actual next period value function onto
            the current period grid of potential next period wealths.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """
    n_choices = options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else range(n_choices)

    next_period_value = np.empty(
        (
            n_choices,
            matrix_next_period_wealth.shape[0] * matrix_next_period_wealth.shape[1],
        )
    )

    for index, state in enumerate(choice_range):
        interpolate_next_period_value = next_period_value_function[state]

        next_period_value[index, :] = interpolate_next_period_value(
            matrix_next_period_wealth.flatten("F"),
        )

    return next_period_value


def get_next_period_policy(
    matrix_next_period_wealth: np.ndarray,
    options: Dict[str, int],
    next_period_policy_function: Dict[int, callable],
) -> np.ndarray:
    """Computes the next-period policy via linear interpolation.

    Extrapolate lineary in wealth regions beyond the grid, i.e. larger
    than "max_wealth" specifiec in the ``params`` dictionary.

    Args:
        policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies of the next period. Dimensions
            of the list are: [n_discrete_choices][2, *n_endog_wealth_grid*], where
            *n_endog_wealth_grid* is of variable length depending on the number of
            concurrent local optima for consumption. The arrays have shape
            [2, *n_endog_wealth_grid*] and are initialized to
            *endog_wealth_grid* = n_grid_wealth + 1. We include one additional
            grid point to the left of the endogenous wealth grid, which we set
            to zero (that's why we have n_grid_wealth + 1 initial points).
            Position [0, :] of the arrays contain the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the (consumption) policy
            function c(M, d), for each time period and each discrete choice.
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        options (dict): Options dictionary.

    Returns:
        next_period_policy (np.ndarray): Array of next period
            consumption of shape (n_choices, n_quad_stochastic * n_grid_wealth).
            Contains interpolated values.
    """
    n_choices = options["n_discrete_choices"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]
    choice_range = [1] if n_choices < 2 else range(n_choices)

    next_period_policy = np.empty((n_choices, n_quad_stochastic * n_grid_wealth))

    for index, state in enumerate(choice_range):
        interpolate_next_period_policy = next_period_policy_function[state]

        next_period_policy[index, :] = interpolate_next_period_policy(
            matrix_next_period_wealth.flatten("F")
        )

    return next_period_policy


def get_current_period_policy(
    state: int,
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
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
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
        state,
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
    state: int,
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
        state (int): State of the agent, e.g. 0 = "retirement", 1 = "working".
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
        state,
        matrix_next_period_wealth,
        next_period_value=next_period_value,
        quad_weights=quad_weights,
        params=params,
        options=options,
    )

    utility = compute_utility(current_period_policy, state, params)
    current_period_value = utility + beta * expected_value

    return expected_value, current_period_value


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
