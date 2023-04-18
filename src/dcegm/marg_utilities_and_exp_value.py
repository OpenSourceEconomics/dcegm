from typing import Callable
from typing import Tuple

import jax.numpy as jnp
from dcegm.interpolation import get_values_and_marginal_utilities
from jax import vmap


def marginal_util_and_exp_max_value_states_period(
    endog_grid_child_states: jnp.ndarray,
    policies_child_states: jnp.ndarray,
    values_child_states: jnp.ndarray,
    exogenous_savings_grid: jnp.ndarray,
    possible_child_states: jnp.ndarray,
    choices_child_states: jnp.ndarray,
    income_shock_draws: jnp.ndarray,
    income_shock_weights: jnp.ndarray,
    taste_shock_scale: float,
    compute_next_period_wealth: Callable,
    compute_marginal_utility: Callable,
    compute_value: Callable,
):
    """Compute the child-state specific marginal utility and expected maximum value.

    The underlying algorithm is the Endogenous-Grid-Method (EGM).

    Args:
        endog_grid_child_states (jnp.ndarray): 3d array containing the endogenous
            wealth grid of the child_states. Shape (n_child_states_period, n_choice,
            n_grid_wealth).
        policies_child_states (jnp.ndarray): 3d array containing the corresponding
            policy function values of the endogenous wealth grid of the child_states
            shape (n_child_states_period, n_choice, n_grid_wealth).
        values_child_states (jnp.ndarray): 3d array containing the corresponding
            value function values of the endogenous wealth grid of the child_states
            shape (n_child_states_period, n_choice, n_grid_wealth).
        exogenous_savings_grid (jnp.array): Exogenous savings grid.
        possible_child_states (jnp.ndarray): Multi-dimensional jnp.ndarray containing
            the possible child_states of shape (n_states_period,num_state_variables).
        choices_child_states (jnp.ndarray): 2d binary jnp.ndarray
            indicating for each child state if choice is possible; of shape
            (n_states_period,num_state_variables).
        income_shock_draws (jnp.array): Stochastic income shock draws.
            Shape (n_stochastic_points).
        income_shock_weights (jnp.array): Weights of stochastic shock draw.
            Shape (n_stochastic_points).
        taste_shock_scale (float): The taste shock scale parameter.
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

    marginal_util_weighted_shock, max_exp_value_weighted_shock = vmap(
        vectorized_marginal_util_and_exp_max_value,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None),
    )(
        endog_grid_child_states,
        resources_next_period,
        policies_child_states,
        values_child_states,
        choices_child_states,
        income_shock_weights,
        taste_shock_scale,
        compute_marginal_utility,
        compute_value,
    )

    return marginal_util_weighted_shock.sum(axis=2), max_exp_value_weighted_shock.sum(
        axis=2
    )


def vectorized_marginal_util_and_exp_max_value(
    endog_grid_child_state: jnp.ndarray,
    next_period_wealth: jnp.ndarray,
    choice_policies_child_state: jnp.ndarray,
    choice_values_child_state: jnp.ndarray,
    choice_set_indices: jnp.ndarray,
    income_shock_weight: jnp.ndarray,
    taste_shock_scale: float,
    compute_marginal_utility: Callable,
    compute_value: Callable,
) -> Tuple[float, float]:
    """Compute the child-state specific marginal utility and expected maximum value.

    The underlying algorithm is the Endogenous-Grid-Method (EGM).

    Args:
        endog_grid_child_state (jnp.ndarray): 2d array containing the endogenous
            wealth grid of each child state. Shape (n_choice, n_grid_wealth).
        next_period_wealth (jnp.ndarray): 1d array of the agent's wealth of the next
            period (t + 1) for each exogenous savings and income shock grid point.
            Shape (n_grid_wealth, n_income_shocks).
        choice_policies_child_state (jnp.ndarray): 2d array containing the corresponding
            policy function values of the endogenous wealth grid of each child state.
            Shape (n_choice, n_grid_wealth).
        choice_values_child_state (jnp.ndarray): 2d array containing the corresponding
            value function values of the endogenous wealth grid of each child state.
            Shape (n_choice, n_grid_wealth).
        choice_set_indices (jnp.ndarray): The agent's (restricted) choice set in the
            given state of shape (n_choices,).
        income_shock_weight (jnp.ndarray): Weights of stochastic shock draws.
        taste_shock_scale (float): The taste shock scale parameter.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_value (callable): User-defined function to compute
            the agent's value function in the credit-constrained area. The inputs
            ```params``` and ```compute_utility``` are already partialled in.

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
        endog_grid_child_state=endog_grid_child_state,
        choice_policies_child_state=choice_policies_child_state,
        choice_values_child_state=choice_values_child_state,
    )

    child_state_marginal_utility_weighted, child_state_exp_max_value_weighted = vmap(
        vmap(
            aggregate_marg_utilites_and_values_over_choices_and_weight,
            in_axes=(1, 1, None, 0, None),
        ),
        in_axes=(1, 1, None, None, None),
    )(
        values_child_state_choice_specific,
        marg_utilities_child_state_choice_specific,
        choice_set_indices,
        income_shock_weight,
        taste_shock_scale,
    )

    return child_state_marginal_utility_weighted, child_state_exp_max_value_weighted


def aggregate_marg_utilites_and_values_over_choices_and_weight(
    values: jnp.ndarray,
    marg_utilities: jnp.ndarray,
    choice_set_indices: jnp.ndarray,
    income_shock_weight: float,
    taste_shock_scale: float,
) -> Tuple[float, float]:
    """Aggregate marginal utilities over discrete choices and weight.

    We aggregate the marginal utilities of the discrete choices in the next period
    over the choice probabilities following from the choice-specific value functions.

    Args:
        values (jnp.ndarray): 1d array of size (n_choices,) containing the agent's
            interpolated values of next period choice-specific value function.
        marg_utilities (jnp.ndarray): 1d array of size (n_choices,) containing the
            agent's interpolated next period policy.
        choice_set_indices (jnp.ndarray): 1d array of the agent's (restricted) choice
            set in the given state of shape (n_choices,).
        income_shock_weight (float): Weight of stochastic shock draw.
        taste_shock_scale (float): Taste shock scale parameter.

    Returns:
        tuple:

        - (float): The marginal utility in the child state.
        - (float): The expected maximum value in the child state.

    """
    values_filtered = jnp.nan_to_num(values, nan=-jnp.inf)
    marg_utilities_filtered = jnp.nan_to_num(marg_utilities, nan=0.0)

    choice_restricted_exp_values, rescale_factor = rescale_values_and_restrict_choices(
        values=values_filtered,
        choice_set_indices=choice_set_indices,
        taste_shock_scale=taste_shock_scale,
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

    # Last step weight outputs
    child_state_marg_util *= income_shock_weight
    child_state_exp_max_value *= income_shock_weight

    return child_state_marg_util, child_state_exp_max_value


def rescale_values_and_restrict_choices(
    values: jnp.ndarray,
    choice_set_indices: jnp.ndarray,
    taste_shock_scale: float,
) -> Tuple[jnp.ndarray, float]:
    """Rescale the choice-restricted values.

    Args:
        values (jnp.ndarray): 1d array of shape (n_choices,) containing the values
            of the next-period choice-specific value function.
        choice_set_indices (jnp.ndarray): 1d array of shape (n_choices,) of the agent's
            (restricted) choice set in the given state.
        taste_shock_scale (float): Taste shock scale parameter.

    Returns:
        tuple:

        - (jnp.ndarray): Rescaled choice-restricted values.
        - (float): Rescaling factor.

    """
    rescale_factor = jnp.amax(values)
    exp_values_scaled = jnp.exp((values - rescale_factor) / taste_shock_scale)
    choice_restricted_exp_values = exp_values_scaled * choice_set_indices

    return choice_restricted_exp_values, rescale_factor
