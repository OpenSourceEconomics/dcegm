"""Implementation of the EGM algorithm."""
from typing import Callable
from typing import Tuple

import numpy as np
from dcegm.interpolate import interpolate_policy
from dcegm.interpolate import interpolate_value


def do_egm_step(
    child_states: np.ndarray,
    state_indexer: np.ndarray,
    state_space: np.ndarray,
    quad_weights: np.ndarray,
    trans_vec_state: np.ndarray,
    taste_shock_scale: float,
    savings_grid: np.ndarray,
    compute_marginal_utility: Callable,
    compute_inverse_marginal_utility: Callable,
    compute_value: Callable,
    compute_next_wealth_matrices: Callable,
    compute_next_marginal_wealth: Callable,
    get_state_specific_choice_set,
    policy_array: np.ndarray,
    value_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs the Endogenous-Grid-Method Algorithm (EGM step).

    Args:
        child_states (np.ndarray): Array of shape (n_exog_processes, n_state_variables)
        capturing the child node for each exogenous process state.
        state_indexer (np.ndarray): Indexer object that maps states to indexes.
            The shape of this object quite complicated. For each state variable it
             has the number of possible states as "row", i.e.
            (n_poss_states_statesvar_1, n_poss_states_statesvar_2, ....)
        state_space (np.ndarray): Collection of all possible states of shape
            (n_states, n_state_variables).
        quad_weights (np.ndarrray): Weights for each stoachstic shock draw.
            Shape is (n_stochastic_quad_points)
        trans_vec_state (np.ndarray): A vector containing for each possible exogenous
            process state the corresponding probability.
            Shape is (n_exog_processes).
        taste_shock_scale (float): The taste shock scale.
        savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            exogenous savings grid .
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_inverse_marginal_utility (callable): User-defined function to compute
        the agent's inverse marginal utility.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
        compute_next_wealth_matrices (callable): User-defined function to compute the
            agent's wealth matrices of the next period (t + 1). The inputs
            ```savings_grid```, ```income_shocks```, ```params``` and ```options```
            are already partialled in.
        compute_next_marginal_wealth (callable): User-defined function to compute the
            agent's marginal wealth in the next period (t + 1). The inputs
            ```params``` and ```options``` are already partialled in.
        get_state_specific_choice_set (Callable): User-supplied function returning for
            each state all possible choices.
        policy_array (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific policy function; of shape
            [n_states, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each state and each discrete choice.
        value_array (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_states, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each state and each discrete choice.

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

    """

    n_exog_process = trans_vec_state.shape[0]
    n_quad_points = quad_weights.shape[0]
    n_savings_grid = savings_grid.shape[0]
    rhs_euler_values = np.empty(
        (
            n_exog_process,
            n_quad_points,
            n_savings_grid,
        ),
        dtype=float,
    )
    max_value_func = np.empty(
        (
            n_exog_process,
            n_quad_points,
            n_savings_grid,
        ),
        dtype=float,
    )

    for i, child_state in enumerate(child_states):
        child_state_index = state_indexer[tuple(child_state)]

        choice_policies_child = policy_array[child_state_index]
        choice_values_child = value_array[child_state_index]

        child_node_choice_set = get_state_specific_choice_set(
            child_state, state_space, state_indexer
        )
        next_period_wealth = compute_next_wealth_matrices(child_state)
        next_period_marginal_wealth = compute_next_marginal_wealth(child_state)

        (
            rhs_euler_values[i, :, :],
            max_value_func[i, :, :],
        ) = get_child_state_policy_and_value(
            child_node_choice_set,
            taste_shock_scale,
            next_period_marginal_wealth,
            compute_marginal_utility,
            compute_value,
            choice_policies_child,
            choice_values_child,
            next_period_wealth,
        )
        rhs_euler_values[i, :, :] *= trans_vec_state[i]
        max_value_func[i, :, :] *= trans_vec_state[i]

    # RHS of Euler Eq., p. 337 IJRS (2017)
    # Integrate out uncertainty over stochastic income y
    rhs_euler = quad_weights @ rhs_euler_values.sum(axis=0)

    expected_value = quad_weights @ max_value_func.sum(axis=0)

    current_policy = compute_inverse_marginal_utility(rhs_euler)

    current_choice = child_states[0][1]

    current_policy_arr, current_value_arr = store_current_period_policy_and_value(
        current_policy,
        expected_value,
        current_choice,
        savings_grid,
        compute_value,
    )

    return current_policy_arr, current_value_arr


def store_current_period_policy_and_value(
    current_policy: np.ndarray,
    expected_value: np.ndarray,
    current_choice: float,
    savings_grid: np.ndarray,
    compute_value,
) -> Tuple[np.ndarray, np.ndarray]:
    """Store the current period policy and value funtions.

    Args:
        current_policy (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the agent's current period policy rule.
        expected_value (np.ndarray): (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the agent's expected value of the next period.
        current_choice (int): The current discrete choice.
        savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            exogenous savings grid .
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.

    Returns:
        (tuple): Tuple containing:

        - current_policy (np.ndarray): 2d array of the agent's period- and
            choice-specific consumption policy. Shape (2, 1.1 * (n_grid_wealth + 1)).
            Position [0, :] contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the policy function c(M, d).
        - current_value (np.ndarray): 2d array of the agent's period- and
            choice-specific value function. Shape (2, 1.1 * (n_grid_wealth + 1)).
            Position [0, :] contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the value function v(M, d).

    """
    n_grid_wealth = savings_grid.shape[0]

    endogenous_wealth_grid = savings_grid + current_policy

    current_policy_container = np.zeros((2, n_grid_wealth + 1))
    current_policy_container[0, 1:] = endogenous_wealth_grid
    current_policy_container[1, 1:] = current_policy

    current_value_container = np.zeros((2, n_grid_wealth + 1))
    current_value_container[0, 1:] = endogenous_wealth_grid
    current_value_container[1, 0] = expected_value[0]
    current_value_container[1, 1:] = compute_value(
        current_policy, expected_value, current_choice
    )

    return current_policy_container, current_value_container


def get_child_state_policy_and_value(
    child_node_choice_set: np.ndarray,
    taste_shock_scale: float,
    next_period_marginal_wealth: np.ndarray,
    compute_marginal_utility: Callable,
    compute_value: Callable,
    choice_policies_child: np.ndarray,
    choice_values_child: np.ndarray,
    next_period_wealth: np.ndarray,
):
    """Runs the Endogenous-Grid-Method Algorithm (EGM step).

    Args:
        child_node_choice_set (np.ndarray): The agent's (restricted) choice set in
            the given state of shape (n_admissible_choices,).
        taste_shock_scale (float): The taste shock scale.
        next_period_marginal_wealth (np.ndarray): The marginal wealth in next period
            with respect to end of current period wealth.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_value (callable): User-defined function to compute
            the agent's value function in the credit-constrained area. The inputs
            ```params``` and ```compute_utility``` are already partialled in.
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
        next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).

    Returns:
    """
    # Interpolate next period policy and values to match the
    # contemporary matrix of potential next period wealths
    child_policy = get_child_state_choice_specific_policy(
        child_node_choice_set,
        next_period_wealth,
        next_period_policy=choice_policies_child,
    )
    choice_child_values = get_child_state_choice_specific_values(
        child_node_choice_set,
        next_period_wealth=next_period_wealth,
        next_period_value=choice_values_child,
        compute_value=compute_value,
    )

    child_state_marginal_utility = get_child_state_policy(
        child_node_choice_set,
        next_period_policy=child_policy,
        next_period_value=choice_child_values,
        taste_shock_scale=taste_shock_scale,
        compute_marginal_utility=compute_marginal_utility,
    ).reshape(next_period_wealth.shape, order="F")

    child_state_log_sum = calc_exp_max_value(
        choice_child_values, taste_shock_scale
    ).reshape(next_period_wealth.shape, order="F")

    child_state_rhs_euler = child_state_marginal_utility * next_period_marginal_wealth

    return child_state_rhs_euler, child_state_log_sum


def calc_exp_max_value(
    choice_specific_values: np.ndarray, taste_shock_scale: float
) -> np.ndarray:
    """Calculate the expected max value given choice specific values. Wit the general
     extrem value assumption on the taste shocks, this reduces to the log-sum.

    The log-sum formula may also be referred to as the 'smoothed max function',
    see eq. (50), p. 335 (Appendix).

    Args:
        choice_specific_values (np.ndarray): Array containing values of the
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        taste_shock_scale (float): Taste shock (scale) parameter.

    Returns:
        logsum (np.ndarray): Log-sum formula inside the expected value function.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
    """
    col_max = np.amax(choice_specific_values, axis=0)
    choice_specific_values_scaled = choice_specific_values - col_max

    # Eq. (14), p. 334 IJRS (2017)
    logsum = col_max + taste_shock_scale * np.log(
        np.sum(np.exp(choice_specific_values_scaled / taste_shock_scale), axis=0)
    )

    return logsum


def get_child_state_policy(
    child_node_choice_set: np.ndarray,
    next_period_policy: np.ndarray,
    next_period_value: np.ndarray,
    compute_marginal_utility: Callable,
    taste_shock_scale: float,
) -> np.ndarray:
    """Aggregates the marginal utility of the discrete choices in the next period with
    the choice probabilities following from the choice-specific value functions.

    Args:
        child_node_choice_set (np.ndarray): 1d array of shape (n_choices_in_state)
            containing the set of all possible choices in the given child state.
        next_period_policy (np.ndarray): 2d array of shape
            (n_choices, n_quad_stochastic * n_grid_wealth) containing the agent's
            interpolated next period policy.
        next_period_value (np.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        compute_marginal_utility (callable): Partial function that calculates marginal
            utility, where the input ```params``` has already been partialed in.
        taste_shock_scale (float): Taste shock (scale) parameter.


    Returns:
        (np.ndarray): Array of next period's marginal utility of shape
            (n_quad_stochastic * n_grid_wealth,).
    """

    next_period_marg_util = np.zeros(next_period_policy.shape[1])

    choice_probabilites = calc_choice_probability(next_period_value, taste_shock_scale)

    for choice_index in range(len(child_node_choice_set)):
        next_period_marg_util += choice_probabilites[
            choice_index
        ] * compute_marginal_utility(next_period_policy[choice_index, :])

    return next_period_marg_util


def get_child_state_choice_specific_policy(
    child_node_choice_set,
    next_period_wealth: np.ndarray,
    next_period_policy: np.ndarray,
) -> np.ndarray:
    """Computes the next-period policy via linear interpolation.

    Extrapolate lineary in wealth regions beyond the grid, i.e. larger
    than "max_wealth" specifiec in the ``params`` dictionary.

    Args:
        child_node_choice_set (np.ndarray): 1d array of shape (n_admissible_choices,)
            containing the agent's choice set at the current child node in the state
            space.
        next_period_wealth (np.ndarray): 2d array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_policy (np.ndarray): Array of the next period policy
            for all choices. Shape (n_choices, 2, 1.1 * n_grid_wealth + 1).
    Returns:
        next_period_policy_interp (np.ndarray): Array of interpolated next period
            consumption of shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """

    next_period_wealth_flat = next_period_wealth.flatten("F")

    next_period_policy_interp = np.empty(
        (child_node_choice_set.shape[0], next_period_wealth_flat.shape[0])
    )

    for index, choice in enumerate(child_node_choice_set):
        next_period_policy_interp[index, :] = interpolate_policy(
            next_period_wealth_flat, next_period_policy[choice]
        )

    return next_period_policy_interp


def get_child_state_choice_specific_values(
    child_node_choice_set: np.ndarray,
    next_period_wealth: np.ndarray,
    next_period_value: np.ndarray,
    compute_value: Callable,
) -> np.ndarray:
    """Maps next-period value onto this period's matrix of next-period wealth.

    Args:
        child_node_choice_set (np.ndarray): 1d array of shape (n_choices_in_state)
            containing the set of all possible choices in the given child state.
        next_period_wealth (np.ndarray): Array of all possible next period
            wealths with shape (n_quad_stochastic, n_grid_wealth).
        next_period_value (np.ndarray): Array of the next period value
            for all choices. Shape (n_choices, 2, 1.1 * n_grid_wealth + 1).
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.

    Returns:
        next_period_value_interp (np.ndarray): Array containing interpolated
            values of next period choice-specific value function. We use
            interpolation to the actual next period value function onto
            the current period grid of potential next period wealths.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
    """

    next_period_wealth_flat = next_period_wealth.flatten("F")
    next_period_value_interp = np.empty(
        (
            child_node_choice_set.shape[0],
            next_period_wealth_flat.shape[0],
        )
    )

    for index, choice in enumerate(child_node_choice_set):
        next_period_value_interp[index, :] = interpolate_value(
            flat_wealth=next_period_wealth_flat,
            value=next_period_value[choice],
            choice=choice,
            compute_value=compute_value,
        )

    return next_period_value_interp


def calc_choice_probability(
    values: np.ndarray,
    taste_shock_scale: float,
) -> np.ndarray:
    """Calculates the probability of working in the next period.

    Args:
        values (np.ndarray): Array containing choice-specific values of the
         value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        taste_shock_scale (float): The taste shock scale.
    Returns:
        prob_working (np.ndarray): Probability of working next period. Array of
            shape (n_quad_stochastic * n_grid_wealth,).
    """
    col_max = np.amax(values, axis=0)
    values_scaled = values - col_max

    # Eq. (15), p. 334 IJRS (2017)
    choice_prob = np.exp(values_scaled / taste_shock_scale) / np.sum(
        np.exp(values_scaled / taste_shock_scale), axis=0
    )

    return choice_prob
