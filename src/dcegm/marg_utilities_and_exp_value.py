from functools import partial
from typing import Callable

import jax.numpy as jnp
import numpy as np
from dcegm.interpolate import interpolate_policy
from dcegm.interpolate import interpolate_value
from jax import jit
from jax import vmap


@partial(jit, static_argnums=(8, 9, 10))
def get_child_state_marginal_util_and_exp_max_value(
    saving: float,
    income_shock: float,
    income_shock_weight: float,
    child_state: np.ndarray,
    choice_set_indices: np.ndarray,
    choice_policies_child: np.ndarray,
    choice_values_child: np.ndarray,
    taste_shock_scale: float,
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
        choice_set_indices (np.ndarray): The agent's (restricted) choice set in the
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

    next_period_wealth = compute_next_period_wealth(child_state, saving, income_shock)
    # Interpolate next period policy and values to match the
    # contemporary matrix of potential next period wealths
    child_policy = vmap(interpolate_policy, in_axes=(None, 0))(
        next_period_wealth, choice_policies_child
    )

    choice_child_values = vmap(interpolate_value, in_axes=(None, 0, None, None))(
        next_period_wealth,
        choice_values_child,
        jnp.arange(choice_values_child.shape[0]),
        compute_value,
    )

    child_state_marginal_utility = get_child_state_marginal_util(
        choice_set_indices=choice_set_indices,
        next_period_policy=child_policy,
        next_period_value=choice_child_values,
        taste_shock_scale=taste_shock_scale,
        compute_marginal_utility=compute_marginal_utility,
    )

    child_state_exp_max_value = calc_exp_max_value(
        choice_child_values, choice_set_indices, taste_shock_scale
    )

    marginal_utility_weighted = child_state_marginal_utility * income_shock_weight

    expected_max_value_weighted = child_state_exp_max_value * income_shock_weight

    return marginal_utility_weighted, expected_max_value_weighted


@partial(jit, static_argnums=(3,))
def get_child_state_marginal_util(
    choice_set_indices,
    next_period_policy: np.ndarray,
    next_period_value: np.ndarray,
    compute_marginal_utility: Callable,
    taste_shock_scale: float,
) -> float:
    """We aggregate the marginal utility of the discrete choices in the next period with
    the choice probabilities following from the choice-specific value functions.

    Args:
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

    choice_probabilities = calc_choice_probability(
        next_period_value, choice_set_indices, taste_shock_scale
    )

    marginal_utility_next_period_policy = vmap(compute_marginal_utility, in_axes=0)(
        next_period_policy
    )
    child_state_marg_util = jnp.sum(
        choice_probabilities * marginal_utility_next_period_policy, axis=0
    )
    return child_state_marg_util


@jit
def calc_choice_probability(
    values: np.ndarray,
    choice_set_indices,
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
    col_max = jnp.amax(values)
    values_scaled = values - col_max

    values_scaled_exp = jnp.exp(values_scaled / taste_shock_scale) * choice_set_indices

    # Eq. (15), p. 334 IJRS (2017)
    choice_prob = values_scaled_exp / jnp.sum(values_scaled_exp)

    return choice_prob


@jit
def calc_exp_max_value(
    choice_specific_values: np.ndarray, choice_set_indices, taste_shock_scale: float
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
    col_max = jnp.amax(choice_specific_values)
    values_scaled = choice_specific_values - col_max

    values_scaled_exp = jnp.exp(values_scaled / taste_shock_scale) * choice_set_indices

    # Eq. (14), p. 334 IJRS (2017)
    logsum = col_max + taste_shock_scale * jnp.log(np.sum(values_scaled_exp))

    return logsum
