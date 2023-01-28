from typing import Callable

import numpy as np
from dcegm.interpolate import interpolate_policy
from dcegm.interpolate import interpolate_value
from jax import vmap


def get_child_state_marginal_util_and_exp_max_value(
    next_period_wealth,
    saving: float,
    income_shock: float,
    income_shock_weight: float,
    child_state: np.ndarray,
    child_node_choice_set: np.ndarray,
    taste_shock_scale: float,
    choice_policies_child: np.ndarray,
    choice_values_child: np.ndarray,
    compute_next_period_wealth: Callable,
    compute_marginal_utility: Callable,
    compute_value: Callable,
):
    """Compute the child-state specific marginal utility and expected maximum value.

    The underlying algorithm is the Endogenous-Grid-Method (EGM).

    Args:
        saving (float): Entry of exogenous savings grid.
        income_shock (float): Entry of income_shock_draws.
        income_shock_weight (float): Weight of stochastic shock draw.
        child_state (np.ndarray): The child state to do calculations for. Shape is
            (n_num_state_variables,).
        child_node_choice_set (np.ndarray): The agent's (restricted) choice set in the
            given state of shape (n_admissible_choices,).
        taste_shock_scale (float): The taste shock scale parameter.
        choice_policies_child (np.ndarray): Multi-dimensional np.ndarray storing the
             corresponding value of the policy function
            c(M, d), for each state and each discrete choice.; of shape
            [n_states, n_discrete_choices, 1.1 * n_grid_wealth + 1].
        choice_values_child (np.ndarray): Multi-dimensional np.ndarray storing the
            corresponding value of the value function
            v(M, d), for each state and each discrete choice; of shape
            [n_states, n_discrete_choices, 1.1 * n_grid_wealth + 1].
        compute_next_period_wealth (callable): User-defined function to compute the
            agent's wealth  of the next period (t + 1). The inputs
            ```saving```, ```income_shock```, ```params``` and ```options```
            are already partialled in.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_value (callable): User-defined function to compute
            the agent's value function in the credit-constrained area. The inputs
            ```params``` and ```compute_utility``` are already partialled in.

    Returns:
        tuple:

        - (np.ndarray): 1d array of the child-state specific marginal utility,
            weighted by the vector of income shocks. Shape (n_grid_wealth,).
        - (np.ndarray): 1d array of the child-state specific expected maximum value,
            weighted by the vector of income shocks. Shape (n_grid_wealth,).

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

    child_state_marginal_utility = get_child_state_marginal_util(
        child_node_choice_set,
        next_period_policy=child_policy,
        next_period_value=choice_child_values,
        taste_shock_scale=taste_shock_scale,
        compute_marginal_utility=compute_marginal_utility,
    )

    child_state_exp_max_value = calc_exp_max_value(
        choice_child_values, taste_shock_scale
    )

    marginal_utility_weighted = child_state_marginal_utility * income_shock_weight

    expected_max_value_weighted = child_state_exp_max_value * income_shock_weight

    return marginal_utility_weighted, expected_max_value_weighted


def get_child_state_marginal_util(
    child_node_choice_set: np.ndarray,
    next_period_policy: np.ndarray,
    next_period_value: np.ndarray,
    compute_marginal_utility: Callable,
    taste_shock_scale: float,
) -> np.ndarray:
    """We aggregate the marginal utility of the discrete choices in the next period with
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
        taste_shock_scale (float): Taste shock scale parameter.


    Returns:
        (float): The marginal utility in the child state.

    """
    child_state_marg_util = 0

    choice_probabilites = calc_choice_probability(next_period_value, taste_shock_scale)

    for choice_index in range(len(child_node_choice_set)):
        child_state_marg_util += choice_probabilites[
            choice_index
        ] * compute_marginal_utility(next_period_policy[choice_index])

    return child_state_marg_util


def get_child_state_choice_specific_policy(
    child_node_choice_set,
    next_period_wealth: float,
    next_period_policy: np.ndarray,
) -> np.ndarray:
    """Compute next-period policy via linear interpolation.

    Extrapolate linearly in wealth regions beyond the grid, i.e. larger
    than "max_wealth", which is specified in the ``params`` dictionary.

    Args:
        child_node_choice_set (np.ndarray): 1d array of shape (n_admissible_choices,)
            containing the agent's choice set at the current child node in the state
            space.
        next_period_wealth (float): possible next period wealth
        next_period_policy (np.ndarray): Array of the next period policy
            for all choices. Shape (n_choices, 2, 1.1 * int(n_grid_wealth) + 1).

    Returns:
        (np.ndarray): 2d array of interpolated next period consumption of shape
            (n_choices, n_quad_stochastic * n_grid_wealth).

    """

    next_period_policy_interp = np.empty(child_node_choice_set.shape[0])

    for index, choice in enumerate(child_node_choice_set):
        next_period_policy_interp[index] = interpolate_policy(
            next_period_wealth, next_period_policy[choice]
        )
    return next_period_policy_interp


def get_child_state_choice_specific_values(
    child_node_choice_set: np.ndarray,
    next_period_wealth: float,
    next_period_value: np.ndarray,
    compute_value: Callable,
) -> np.ndarray:
    """Map next-period value onto this period's matrix of next-period wealth.

    Args:
        child_node_choice_set (np.ndarray): 1d array of shape (n_choices_in_state)
            containing the set of all possible choices in the given child state.
        next_period_wealth (float): possible next period wealth
        next_period_value (np.ndarray): Array of the next period value
            for all choices. Shape (n_choices, 2, 1.1 * n_grid_wealth + 1).
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.

    Returns:
        (np.ndarray): Array containing interpolated values of next period
            choice-specific value function. We use interpolation to the actual next
            period value function onto the current period grid of potential next
            period wealths. Shape (n_choices, n_quad_stochastic * n_grid_wealth).

    """

    next_period_value_interp = np.empty(child_node_choice_set.shape[0])

    for index, choice in enumerate(child_node_choice_set):
        next_period_value_interp[index] = interpolate_value(
            flat_wealth=next_period_wealth,
            value=next_period_value[choice],
            choice=choice,
            compute_value=compute_value,
        )
    return next_period_value_interp


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
    col_max = np.amax(values)
    values_scaled = values - col_max

    # Eq. (15), p. 334 IJRS (2017)
    choice_prob = np.exp(values_scaled / taste_shock_scale) / np.sum(
        np.exp(values_scaled / taste_shock_scale)
    )

    return choice_prob


def calc_exp_max_value(
    choice_specific_values: np.ndarray, taste_shock_scale: float
) -> np.ndarray:
    """Calculate the expected maximum value given choice specific values.

    With the general extreme value assumption on the taste shocks, this reduces
    to the log-sum.

    The log-sum formula may also be referred to as the 'smoothed max function',
    see eq. (50), p. 335 (Appendix).

    Args:
        choice_specific_values (np.ndarray): Array containing values of the
            choice-specific value function.
            Shape (n_choices, n_quad_stochastic * n_grid_wealth).
        taste_shock_scale (float): Taste shock scale parameter.

    Returns:
        (np.ndarray): Log-sum formula inside the expected value function.
            2d array of shape (n_quad_stochastic * n_grid_wealth,).

    """
    col_max = np.amax(choice_specific_values)
    choice_specific_values_scaled = choice_specific_values - col_max

    # Eq. (14), p. 334 IJRS (2017)
    logsum = col_max + taste_shock_scale * np.log(
        np.sum(np.exp(choice_specific_values_scaled / taste_shock_scale))
    )

    return logsum
