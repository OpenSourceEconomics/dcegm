from typing import Callable
from typing import Tuple

import jax.numpy as jnp
from dcegm.interpolate import get_values_and_marginal_utilities
from jax import vmap


def marginal_util_and_exp_max_value_states_period(
    compute_next_period_wealth: Callable,
    compute_marginal_utility: Callable,
    compute_value: Callable,
    taste_shock_scale: float,
    exogenous_savings_grid: jnp.ndarray,
    income_shock_draws: jnp.ndarray,
    income_shock_weights: jnp.ndarray,
    possible_child_states: jnp.ndarray,
    choices_child_states: jnp.ndarray,
    engog_grid_child_states: jnp.ndarray,
    policies_child_states: jnp.ndarray,
    values_child_states: jnp.ndarray,
):
    """Compute the child-state specific marginal utility and expected maximum value.

    The underlying algorithm is the Endogenous-Grid-Method (EGM).

    Args:
        compute_next_period_wealth (callable): User-defined function to compute the
            agent's wealth  of the next period (t + 1). The inputs
            ```saving```, ```income_shock```, ```params``` and ```options```
            are already partialled in.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_value (callable): User-defined function to compute
            the agent's value function in the credit-constrained area. The inputs
            ```params``` and ```compute_utility``` are already partialled in.
        taste_shock_scale (float): The taste shock scale parameter.
        exogenous_savings_grid (jnp.array): Exogenous savings grid.
        income_shock_draws (jnp.array): Stochastic income shock draws.
            Shape (n_stochastic_points).
        income_shock_weights (jnp.array): Weights of stochastic shock draw.
            Shape (n_stochastic_points).
        possible_child_states (jnp.ndarray): Multi-dimensional jnp.ndarray containing
            the possible child_states; of shape (n_states_period,num_state_variables).
        choices_child_states (jnp.ndarray): Multi-dimensional binary jnp.ndarray
            indicating for each child state if choice is possible; of shape
            (n_states_period,num_state_variables).
        policies_child_states (jnp.ndarray): Multi-dimensional jnp.ndarray storing the
             corresponding value of the policy function c(M, d), for each child state
             and each discrete choice; of shape
            [n_states_period, n_discrete_choices, 1.1 * n_grid_wealth + 1].
        values_child_states (jnp.ndarray): Multi-dimensional jnp.ndarray storing the
            corresponding value of the value function v(M,d), for each child state
            and each discrete choice; of shape
            [n_states_period, n_discrete_choices, 1.1 * n_grid_wealth + 1].

    Returns:
        tuple:
        - (jnp.ndarray): 1d array of the child-state specific marginal utility,
            weighted by the vector of income shocks.
            Shape (n_states_period, n_grid_wealth).
        - (jnp.ndarray): 1d array of the child-state specific expected maximum value,
            weighted by the vector of income shocks.
            Shape (n_states_period, n_grid_wealth).

    """
    resources_next_period = vmap(
        vmap(
            vmap(compute_next_period_wealth, in_axes=(None, None, 0)),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, None, None),
    )(possible_child_states, exogenous_savings_grid, income_shock_draws)

    (
        marginal_util_weighted_shock,
        max_exp_value_weighted_shock,
    ) = vmap(
        vmap(
            vmap(
                vectorized_marginal_util_and_exp_max_value,
                in_axes=(None, None, None, 0, 0, None, None, None, None),
            ),
            in_axes=(None, None, None, None, 0, None, None, None, None),
        ),
        in_axes=(None, None, None, None, 0, 0, 0, 0, 0),
    )(
        compute_marginal_utility,
        compute_value,
        taste_shock_scale,
        income_shock_weights,
        resources_next_period,
        engog_grid_child_states,
        choices_child_states,
        policies_child_states,
        values_child_states,
    )

    return marginal_util_weighted_shock.sum(axis=2), max_exp_value_weighted_shock.sum(
        axis=2
    )


def vectorized_marginal_util_and_exp_max_value(
    compute_marginal_utility: Callable,
    compute_value: Callable,
    taste_shock_scale: float,
    income_shock_weight: float,
    next_period_wealth,
    engog_grid_child_states,
    choice_set_indices: jnp.ndarray,
    choice_policies_child: jnp.ndarray,
    choice_values_child: jnp.ndarray,
) -> Tuple[float, float]:
    """Compute the child-state specific marginal utility and expected maximum value.

    The underlying algorithm is the Endogenous-Grid-Method (EGM).

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_value (callable): User-defined function to compute
            the agent's value function in the credit-constrained area. The inputs
            ```params``` and ```compute_utility``` are already partialled in.
        taste_shock_scale (float): The taste shock scale parameter.
        income_shock_weight (float): Weight of stochastic shock draw.
        choice_set_indices (jnp.ndarray): The agent's (restricted) choice set in the
            given state of shape (n_choices,).
        choice_policies_child (jnp.ndarray): Multi-dimensional jnp.ndarray storing the
             corresponding value of the policy function c(M, d), for each state and
             each discrete choice; of shape
            [n_states, n_discrete_choices, 1.1 * n_grid_wealth + 1].
        choice_values_child (jnp.ndarray): Multi-dimensional jnp.ndarray storing the
            corresponding value of the value function
            v(M, d), for each state and each discrete choice; of shape
            [n_states, n_discrete_choices, 1.1 * n_grid_wealth + 1].

    Returns:
        tuple:

        - (float): 1d array of the child-state specific marginal utility,
            weighted by the vector of income shocks. Shape (n_grid_wealth,).
        - (jnp.ndarray): 1d array of the child-state specific expected maximum value,
            weighted by the vector of income shocks. Shape (n_grid_wealth,).

    """

    # Interpolate the optimal consumption choice on the wealth grid and calculate the
    # corresponding marginal utilities.
    (
        marg_utilities_child_state_choice_specific,
        values_child_state_choice_specific,
    ) = get_values_and_marginal_utilities(
        compute_marginal_utility=compute_marginal_utility,
        compute_value=compute_value,
        next_period_wealth=next_period_wealth,
        choice_policies_child=choice_policies_child,
        value_functions_child=choice_values_child,
        endog_grid=engog_grid_child_states,
    )

    (
        child_state_marginal_utility_weighted,
        child_state_exp_max_value_weighted,
    ) = aggregate_marg_utilites_and_values_over_choices_and_weight(
        choice_set_indices=choice_set_indices,
        marg_utilities=marg_utilities_child_state_choice_specific,
        values=values_child_state_choice_specific,
        income_shock_weight=income_shock_weight,
        taste_shock_scale=taste_shock_scale,
    )

    return child_state_marginal_utility_weighted, child_state_exp_max_value_weighted


def aggregate_marg_utilites_and_values_over_choices_and_weight(
    choice_set_indices: jnp.ndarray,
    marg_utilities: jnp.ndarray,
    values: jnp.ndarray,
    income_shock_weight: float,
    taste_shock_scale: float,
) -> Tuple[float, float]:
    """We aggregate the marginal utility of the discrete choices in the next period with
    the choice probabilities following from the choice-specific value functions.

    Args:
        choice_set_indices (jnp.ndarray): The agent's (restricted) choice set in the
            given state of shape (n_choices,).
        marg_utilities (jnp.ndarray): 1d array of size (n_choices) containing the
            agent's interpolated next period policy.
        values (jnp.ndarray): 1d array of size (n_choices) containing the agent's
            interpolated values of next period choice-specific value function.
        income_shock_weight (float): Weight of stochastic shock draw.
        taste_shock_scale (float): Taste shock scale parameter.

    Returns:
        (float): The marginal utility in the child state.
        (float): The expected maximum value in the child state.

    """
    values_filtered = jnp.nan_to_num(values, nan=-jnp.inf)
    marg_utilities_filtered = jnp.nan_to_num(marg_utilities, nan=0.0)

    choice_restricted_exp_values, rescale_factor = rescale_values_and_restrict_choices(
        values_filtered, taste_shock_scale, choice_set_indices
    )

    sum_exp_values = jnp.sum(choice_restricted_exp_values, axis=0)

    # Compute the choice probabilities
    choice_probabilities = choice_restricted_exp_values / sum_exp_values
    # Aggregate marginal utilities with choice probabilities and filter before
    child_state_marg_util = jnp.sum(
        choice_probabilities * marg_utilities_filtered, axis=0
    )
    # Calculate the expected maximum value with the log-sum formula
    child_state_exp_max_value = rescale_factor + taste_shock_scale * jnp.log(
        sum_exp_values
    )

    # Last step weight outpouts
    child_state_marg_util *= income_shock_weight
    child_state_exp_max_value *= income_shock_weight

    return child_state_marg_util, child_state_exp_max_value


def rescale_values_and_restrict_choices(
    values: jnp.ndarray, taste_shock_scale: float, choice_set_indices: jnp.ndarray
) -> Tuple[jnp.ndarray, float]:
    """Rescale the choice-restricted values.

    Args:
        values (jnp.ndarray): Array containing values of next period
            choice-specific value function; of shape (n_choices,).
        taste_shock_scale (float): Taste shock scale parameter.
        choice_set_indices (jnp.ndarray): The agent's (restricted) choice set in the
            given state of shape (n_choices,).

    Returns:
        (jnp.ndarray): Rescaled choice-restricted values.
        (float): Rescaling factor.

    """
    rescale_factor = jnp.amax(values)
    exp_values_scaled = jnp.exp((values - rescale_factor) / taste_shock_scale)
    choice_restricted_exp_values = exp_values_scaled * choice_set_indices

    return choice_restricted_exp_values, rescale_factor
