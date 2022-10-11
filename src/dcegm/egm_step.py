"""Implementation of the EGM algorithm."""
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
from dcegm.interpolate import interpolate_policy
from dcegm.interpolate import interpolate_value


def do_egm_step(
    child_state,
    child_node_choice_set,
    quad_weights,
    *,
    options: Dict[str, int],
    compute_utility: Callable,
    compute_marginal_utility: Callable,
    compute_inverse_marginal_utility: Callable,
    compute_value_constrained: Callable,
    compute_expected_value: Callable,
    compute_next_choice_probs: Callable,
    compute_next_wealth_matrices: Callable,
    compute_next_marginal_wealth: Callable,
    store_current_policy_and_value: Callable,
    choice_policies_child: np.ndarray,
    choice_values_child: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the Endogenous-Grid-Method Algorithm (EGM step).

    Args:
        child_state (np.ndarray): Array of shape (n_state_variables,) defining the
            agent's current child state.
        child_node_choice_set (np.ndarray): The agent's (restricted) choice set in
            the given state of shape (n_admissible_choices,).
        options (dict): Options dictionary.
        compute_utility (callable): User-defined function to compute the agent's
            utility. The input ```params``` is already partialled in.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_inverse_marginal_utility (callable): User-defined function to compute
        the agent's inverse marginal utility.
        compute_value_constrained (callable): User-defined function to compute
            the agent's value function in the credit-constrained area. The inputs
            ```params``` and ```compute_utility``` are already partialled in.
        compute_expected_value (callable): User-defined function to compute the agent's
            expected value. The inputs ```params``` and ```quad_weights``` are already
            partialled in.
        compute_next_choice_probs (callable): User-defined function to compute the
            agent's choice probabilities in the next period (t + 1). The inputs
            ```params``` and ```options``` are already partialled in.
        compute_next_wealth_matrices (callable): User-defined function to compute the
            agent's wealth matrices of the next period (t + 1). The inputs
            ```savings_grid```, ```income_shocks```, ```params``` and ```options```
            are already partialled in.
        store_current_policy_and_value (callable): Internal function that computes the
            current state- and choice-specific optimal policy and value functions.
            The inputs ```savings_grid```, ```params```, ```options```, and
            ```compute_utility``` are already partialled in.
        compute_next_marginal_wealth (callable): User-defined function to compute the
            agent's marginal wealth in the next period (t + 1). The inputs
            ```params``` and ```options``` are already partialled in.
        choice_policies_child (np.ndarray): 2d array of the agent's next period policy
            for all choices. Shape (n_choices, 2, 1.1 * (n_grid_wealth + 1)).
            Position [:, 0, :] contains the endogenous grid over wealth M,
            and [:, 1, :] stores the corresponding value of the choice-specific policy
            function c(M, d).
        choice_values_child (np.ndarray): 2d array of the agent's next period values
            for all choices. Shape (n_choices, 2, 1.1 * (n_grid_wealth + 1)).
            Position [:, 0, :] contains the endogenous grid over wealth M,
            and [:, 1, :] stores the corresponding value of the choice-specific value
            function v(M, d).

    Returns:
        (tuple) Tuple containing:

        - current_policy (np.ndarray): 2d array of the agent's period- and
            choice-specific consumption policy. Shape (2, 1.1 * (n_grid_wealth + 1)).
            Position [0, :] contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the policy function c(M, d).
        - current_value (np.ndarray): 2d array of the agent's period- and
            choice-specific value function. Shape (2, 1.1 * (n_grid_wealth + 1)).
            Position [0, :] contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the value function v(M, d).
        - expected_value (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the agent's expected value of the next period.

    """
    next_period_wealth = compute_next_wealth_matrices(child_state)
    next_period_marginal_wealth = compute_next_marginal_wealth(child_state)

    next_period_marginal_utility, next_period_values = get_child_state_policy_and_value(
        child_state,
        child_node_choice_set,
        options,
        compute_utility,
        compute_marginal_utility,
        compute_value_constrained,
        compute_next_choice_probs,
        choice_policies_child,
        choice_values_child,
        next_period_wealth,
    )

    # RHS of Euler Eq., p. 337 IJRS (2017)
    # Integrate out uncertainty over stochastic income y
    rhs_euler = quad_weights @ (
        next_period_marginal_utility * next_period_marginal_wealth
    )

    current_policy = compute_inverse_marginal_utility(rhs_euler)

    expected_value = compute_expected_value(next_period_wealth, next_period_values)

    current_policy_arr, current_value_arr = store_current_policy_and_value(
        current_policy, expected_value, child_state
    )

    return current_policy_arr, current_value_arr, expected_value


# def aggregate_over_income_shocks():


def get_child_state_policy_and_value(
    child_state: np.ndarray,
    child_node_choice_set: np.ndarray,
    options: Dict[str, int],
    compute_utility: Callable,
    compute_marginal_utility: Callable,
    compute_value_constrained: Callable,
    compute_next_choice_probs: Callable,
    choice_policies_child: np.ndarray,
    choice_values_child: np.ndarray,
    next_period_wealth: np.ndarray,
):
    """Runs the Endogenous-Grid-Method Algorithm (EGM step).

    Args:
        child_state (np.ndarray): Array of shape (n_state_variables,) defining the
            agent's current child state.
        child_node_choice_set (np.ndarray): The agent's (restricted) choice set in
            the given state of shape (n_admissible_choices,).
        options (dict): Options dictionary.
        compute_utility (callable): User-defined function to compute the agent's
            utility. The input ```params``` is already partialled in.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_current_policy (callable): User-defined function to compute the agent's
            current state- and choice-specific optimal policy. The inputs
            ```quad_weights```and ```compute_inverse_margina_utility``` are already
            partialled in.
        compute_value_credit_constrained (callable): User-defined function to compute
            the agent's value function in the credit-constrained area. The inputs
            ```params``` and ```compute_utility``` are already partialled in.
        compute_expected_value (callable): User-defined function to compute the agent's
            expected value. The inputs ```params``` and ```quad_weights``` are already
            partialled in.
        compute_next_choice_probs (callable): User-defined function to compute the
            agent's choice probabilities in the next period (t + 1). The inputs
            ```params``` and ```options``` are already partialled in.
        compute_next_wealth_matrices (callable): User-defined function to compute the
            agent's wealth matrices of the next period (t + 1). The inputs
            ```savings_grid```, ```income_shocks```, ```params``` and ```options```
            are already partialled in.
        store_current_policy_and_value (callable): Internal function that computes the
            current state- and choice-specific optimal policy and value functions.
            The inputs ```savings_grid```, ```params```, ```options```, and
            ```compute_utility``` are already partialled in.
        compute_next_marginal_wealth (callable): User-defined function to compute the
            agent's marginal wealth in the next period (t + 1). The inputs
            ```params``` and ```options``` are already partialled in.
        choice_policies_child (np.ndarray): 2d array of the agent's next period policy
            for all choices. Shape (n_choices, 2, 1.1 * (n_grid_wealth + 1)).
            Position [:, 0, :] contains the endogenous grid over wealth M,
            and [:, 1, :] stores the corresponding value of the choice-specific policy
            function c(M, d).
        choice_values_child (np.ndarray): 2d array of the agent's next period values
            for all choices. Shape (n_choices, 2, 1.1 * (n_grid_wealth + 1)).
            Position [:, 0, :] contains the endogenous grid over wealth M,
            and [:, 1, :] stores the corresponding value of the choice-specific value
            function v(M, d).

    Returns:
    """
    # Interpolate next period policy and values to match the
    # contemporary matrix of potential next period wealths
    child_policy = get_next_period_policy(
        child_node_choice_set,
        next_period_wealth,
        next_period_policy=choice_policies_child,
        options=options,
    )
    child_value = get_next_period_value(
        child_node_choice_set,
        matrix_next_period_wealth=next_period_wealth,
        period=child_state[0] - 1,
        options=options,
        next_period_value=choice_values_child,
        compute_value_constrained=compute_value_constrained,
        compute_utility=compute_utility,
    )

    child_state_marginal_utility = sum_marginal_utility_over_choice_probs(
        child_node_choice_set,
        next_period_policy=child_policy,
        next_period_value=child_value,
        options=options,
        compute_marginal_utility=compute_marginal_utility,
        compute_next_period_choice_probs=compute_next_choice_probs,
    )
    return child_state_marginal_utility, child_value


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
