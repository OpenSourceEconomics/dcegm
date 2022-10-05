"""Implementation of the EGM algorithm."""
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
from scipy import interpolate


def do_egm_step(
    child_state,
    child_node_choice_set,
    *,
    options: Dict[str, int],
    compute_utility: Callable,
    compute_marginal_utility: Callable,
    compute_current_policy: Callable,
    compute_value_constrained: Callable,
    compute_expected_value: Callable,
    compute_next_choice_probs: Callable,
    compute_next_wealth_matrices: Callable,
    compute_next_marginal_wealth: Callable,
    store_current_policy_and_value: Callable,
    next_policy: np.ndarray,
    next_value: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the Endogenous-Grid-Method Algorithm (EGM step).

    Args:
        child_state (np.ndarray): Current individual child state.
        options (dict): Options dictionary.
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of (i) utility, (ii) inverse marginal utility,
            and (iii) next period marginal utility. All three are partial functions,
            where the common input ```params``` has already been partialled in.
        compute_income (callable): User-defined function to calculate the agent's
            end-of-period (labor) income, where the inputs ```quad_points```,
            ```params``` and ```options``` are already partialled in.
        compute_value_credit_constrained (callable): User-defined function to compute
            the agent's value function in the credit-constrained area.
            The inputs ```params``` and ```compute_utility``` are already partialled in.
        next_period_policy (np.ndarray): Array of the next period policy
            for all choices. Shape (n_choices, 2, 1.1 * (n_grid_wealth + 1)).
        next_period_value (np.ndarray): Array of the next period values
            for all choices. Shape (n_choices, 2, 1.1 * (n_grid_wealth + 1)).

    Returns:
        (tuple) Tuple containing

        - current_policy (np.ndarray): Nested list of np.ndarrays storing the
            choice-specific consumption policies. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*].
        - current_value (np.ndarray): Nested list of np.ndarrays storing the
            choice-specific value functions. Dimensions of the list are:
            [n_periods][n_discrete_choices][2, *n_endog_wealth_grid*].
        - expected_value (np.ndarray): The expected value of continuation.

    """
    next_wealth = compute_next_wealth_matrices(child_state)
    next_marginal_wealth = compute_next_marginal_wealth(child_state)

    # Interpolate next period policy and values to match the
    # contemporary matrix of potential next period wealths
    next_policy = get_next_period_policy(
        child_node_choice_set,
        next_wealth,
        next_period_policy=next_policy,
        options=options,
    )
    next_value = get_next_period_value(
        child_node_choice_set,
        matrix_next_period_wealth=next_wealth,
        period=child_state[0] - 1,
        options=options,
        next_period_value=next_value,
        compute_value_constrained=compute_value_constrained,
        compute_utility=compute_utility,
    )

    next_marginal_utility = sum_marginal_utility_over_choice_probs(
        child_node_choice_set,
        next_period_policy=next_policy,
        next_period_value=next_value,
        options=options,
        compute_marginal_utility=compute_marginal_utility,
        compute_next_period_choice_probs=compute_next_choice_probs,
    )

    current_policy = compute_current_policy(next_marginal_utility, next_marginal_wealth)
    expected_value = compute_expected_value(next_wealth, next_value)

    current_policy_arr, current_value_arr = store_current_policy_and_value(
        current_policy, expected_value, child_state
    )

    return current_policy_arr, current_value_arr, expected_value


def sum_marginal_utility_over_choice_probs(
    child_node_choice_set: np.ndarray,
    next_period_policy: np.ndarray,
    next_period_value: np.ndarray,
    options: dict,
    compute_marginal_utility: Callable,
    compute_next_period_choice_probs: Callable,
) -> np.ndarray:
    """Computes the marginal utility of the next period.

    Args:
        child_node_choice_set (np.ndarray): 1d array of shape (n_choices_in_state)
            containing the set of all possible choices in the given child state.
        marginal_utility_func (callable): Partial function that calculates marginal
            utility, where the input ```params``` has already been partialed in.
            Supposed to have same interface as utility func.
        next_period_policy (np.ndarray): 2d array of shape
            (n_choices, n_quad_stochastic * n_grid_wealth) containing the agent's
            interpolated next period policy.
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        options (dict): Options dictionary.

    Returns:
        (np.ndarray): Array of next period's marginal utility of shape
            (n_quad_stochastic * n_grid_wealth,).
    """
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]

    next_period_marg_util = np.zeros(next_period_policy.shape[1])

    for choice_index in range(len(child_node_choice_set)):
        choice_prob = compute_next_period_choice_probs(next_period_value, choice_index)
        next_period_marg_util += choice_prob * compute_marginal_utility(
            next_period_policy[choice_index, :]
        )

    return next_period_marg_util.reshape((n_quad_stochastic, n_grid_wealth), order="F")


def get_next_period_policy(
    child_node_choice_set,
    matrix_next_period_wealth: np.ndarray,
    next_period_policy: np.ndarray,
    options: Dict[str, int],
) -> np.ndarray:
    """Computes the next-period policy via linear interpolation.

    Extrapolate lineary in wealth regions beyond the grid, i.e. larger
    than "max_wealth" specifiec in the ``params`` dictionary.

    Args:
        child_node_choice_set (np.ndarray): 1d array of shape (n_admissible_choices,)
            containing the agent's choice set at the current child node in the state
            space.
        matrix_next_period_wealth (np.ndarray): 2d array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_policy (np.ndarray): Array of the next period policy
            for all choices. Shape (n_choices, 2, 1.1 * n_grid_wealth + 1).
        options (dict): Options dictionary.

    Returns:
        next_period_policy_interp (np.ndarray): Array of interpolated next period
            consumption of shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_stochastic = options["quadrature_points_stochastic"]

    next_period_policy_interp = np.empty(
        (child_node_choice_set.shape[0], n_quad_stochastic * n_grid_wealth)
    )

    for index, choice in enumerate(child_node_choice_set):
        next_period_policy_interp[index, :] = interpolate_policy(
            matrix_next_period_wealth.flatten("F"), next_period_policy[choice]
        )

    return next_period_policy_interp


def get_next_period_value(
    child_node_choice_set,
    matrix_next_period_wealth: np.ndarray,
    next_period_value: np.ndarray,
    period: int,
    options: Dict[str, int],
    compute_utility: Callable,
    compute_value_constrained: Callable,
) -> np.ndarray:
    """Maps next-period value onto this period's matrix of next-period wealth.

    Args:
        matrix_next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_value (np.ndarray): Array of the next period value
            for all choices. Shape (n_choices, 2, 1.1 * n_grid_wealth + 1).
        period (int): Current period t.
        options (dict): Options dictionary.
        compute_value_credit_constrained (callable): User-defined function to compute
            the agent's utility. The input ```params``` is already partialled in.
        compute_value_credit_constrained (callable): User-defined function to compute
            the agent's value function in the credit-constrained area.
            The inputs ```params``` and ```compute_utility``` are already partialled in.

    Returns:
        next_period_value_interp (np.ndarray): Array containing interpolated
            values of next period choice-specific value function. We use
            interpolation to the actual next period value function onto
            the current period grid of potential next period wealths.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """
    next_period_value_interp = np.empty(
        (
            child_node_choice_set.shape[0],
            matrix_next_period_wealth.shape[0] * matrix_next_period_wealth.shape[1],
        )
    )

    for index, choice in enumerate(child_node_choice_set):
        if period == options["n_periods"] - 2:
            next_period_value_interp[index, :] = compute_utility(
                matrix_next_period_wealth.flatten("F"), choice
            )
        else:
            next_period_value_interp[index, :] = interpolate_value(
                flat_wealth=matrix_next_period_wealth.flatten("F"),
                value=next_period_value[choice],
                choice=choice,
                compute_value_constrained=compute_value_constrained,
            )

    return next_period_value_interp


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
    compute_value_constrained: Callable,
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

    Returns:
        (np.ndarray): Interpolated flat value function of shape
            (n_quad_stochastic * n_grid_wealth,).
    """
    value = value[:, ~np.isnan(value).any(axis=0)]
    value_interp = np.empty(flat_wealth.shape)

    # Mark credit constrained region
    constrained_region = flat_wealth < value[0, 1]

    # Calculate t+1 value function in constrained region using
    # the analytical part
    value_interp[constrained_region] = compute_value_constrained(
        flat_wealth[constrained_region],
        next_period_value=value[1, 0],
        choice=choice,
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
