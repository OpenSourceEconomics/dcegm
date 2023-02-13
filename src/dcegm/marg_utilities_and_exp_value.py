from functools import partial
from typing import Callable
from typing import Tuple

import jax.numpy as jnp
from dcegm.interpolate import interpolate_and_calc_marginal_utilities
from dcegm.interpolate import interpolate_value
from jax import jit
from jax import vmap


@partial(jit, static_argnums=(8, 9, 10))
def get_child_state_marginal_util_and_exp_max_value(
    saving: float,
    income_shock: float,
    income_shock_weight: float,
    child_state: jnp.ndarray,
    choice_set_indices: jnp.ndarray,
    choice_policies_child: jnp.ndarray,
    choice_values_child: jnp.ndarray,
    taste_shock_scale: float,
    compute_next_period_wealth: Callable,
    compute_marginal_utility: Callable,
    compute_value: Callable,
) -> Tuple[float, float]:
    """Compute the child-state specific marginal utility and expected maximum value.

    The underlying algorithm is the Endogenous-Grid-Method (EGM).

    Args:
        saving (float): Entry of exogenous savings grid.
        income_shock (float): Entry of income_shock_draws.
        income_shock_weight (float): Weight of stochastic shock draw.
        child_state (jnp.ndarray): The child state to do calculations for. Shape is
            (n_num_state_variables,).
        choice_set_indices (jnp.ndarray): The agent's (restricted) choice set in the
            given state of shape (n_admissible_choices,).
        taste_shock_scale (float): The taste shock scale parameter.
        choice_policies_child (jnp.ndarray): Multi-dimensional jnp.ndarray storing the
             corresponding value of the policy function
            c(M, d), for each state and each discrete choice.; of shape
            [n_states, n_discrete_choices, 1.1 * n_grid_wealth + 1].
        choice_values_child (jnp.ndarray): Multi-dimensional jnp.ndarray storing the
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

        - (float): 1d array of the child-state specific marginal utility,
            weighted by the vector of income shocks. Shape (n_grid_wealth,).
        - (jnp.ndarray): 1d array of the child-state specific expected maximum value,
            weighted by the vector of income shocks. Shape (n_grid_wealth,).

    """

    next_period_wealth = compute_next_period_wealth(state=child_state, saving=saving, income_shock=income_shock)

    marg_utilities_child_state_choice_specific = (
        interpolate_and_calc_marginal_utilities(
            compute_marginal_utility=compute_marginal_utility,
            next_period_wealth=next_period_wealth,
            choice_policies_child=choice_policies_child,
        )
    )

    values_child_state_choice_specific = vmap(
        interpolate_value, in_axes=(None, 0, None, None)
    )(
        next_period_wealth,
        choice_values_child,
        jnp.arange(choice_values_child.shape[0]),
        compute_value,
    )

    (
        child_state_marginal_utility,
        child_state_exp_max_value,
    ) = aggregate_marg_utilites_and_values_over_choices(
        choice_set_indices=choice_set_indices,
        marg_utilities=marg_utilities_child_state_choice_specific,
        values=values_child_state_choice_specific,
        taste_shock_scale=taste_shock_scale,
    )

    marginal_utility_weighted = child_state_marginal_utility * income_shock_weight

    expected_max_value_weighted = child_state_exp_max_value * income_shock_weight

    return marginal_utility_weighted, expected_max_value_weighted


@jit
def aggregate_marg_utilites_and_values_over_choices(
    choice_set_indices: jnp.ndarray,
    marg_utilities: jnp.ndarray,
    values: jnp.ndarray,
    taste_shock_scale: float,
) -> Tuple[float, float]:
    """We aggregate the marginal utility of the discrete choices in the next period with
    the choice probabilities following from the choice-specific value functions.

    Args:
        marg_utilities (jnp.ndarray): 1d array of size (n_choices) containing the
            agent's interpolated next period policy.
        values (jnp.ndarray): 1d array of size (n_choices) containing the agent's
            interpolated values of next period choice-specific value function.
        taste_shock_scale (float): Taste shock scale parameter.


    Returns:
        (float): The marginal utility in the child state.
        (float): The expected maximum value in the child state.

    """
    choice_restricted_exp_values, rescale_factor = rescale_values_and_restrict_choices(
        values, taste_shock_scale, choice_set_indices
    )

    sum_exp_values = jnp.sum(choice_restricted_exp_values, axis=0)

    # Compute the choice probabilities
    choice_probabilities = choice_restricted_exp_values / sum_exp_values
    # Aggregate marginal utilities with choice probabilities
    child_state_marg_util = jnp.sum(choice_probabilities * marg_utilities, axis=0)
    # Calculate the expected maximum value with the log-sum formula
    child_state_exp_max_value = rescale_factor + taste_shock_scale * jnp.log(
        sum_exp_values
    )

    return child_state_marg_util, child_state_exp_max_value


@jit
def rescale_values_and_restrict_choices(
    values: jnp.ndarray, taste_shock_scale: float, choice_set_indices: jnp.ndarray
) -> Tuple[jnp.ndarray, float]:
    """Rescale the choice-restricted values.

    Args:
        values (jnp.ndarray): Array containing values of next period
            choice-specific value function.
            Shape (n_choices).
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
