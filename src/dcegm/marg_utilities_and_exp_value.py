from typing import Tuple

import jax.numpy as jnp
from jax import numpy as jnp


def aggregate_marg_utils_exp_values(
    value_state_choice_combs: jnp.ndarray,
    marg_util_state_choice_combs: jnp.ndarray,
    reshape_state_choice_vec_to_mat: jnp.ndarray,
    sum_state_choice_to_state: jnp.ndarray,
    taste_shock_scale: float,
    income_shock_weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the aggregate marginal utilities and expected values.

    Args:
        value_state_choices (jnp.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the value function for all
            state-choice combinations and income shocks.
        marg_util_state_choices (jnp.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the marginal utility of
            consumption for all states, choices, and income shocks.
        sum_state_choices_to_state (jnp.ndarray): 2d array of shape
            (n_states, n_state_choices) with state_space size with ones, where
            state-choice belongs to state.
        taste_shock_scale (float): The taste shock scale.
        income_shock_weights (jnp.ndarray): 1d array of shape (n_stochastic_quad_points)

    Returns:
        tuple:

        - marginal_utils_draws (np.ndarray): 1d array of shape
        (n_states, n_savings, n_income_shocks,) of the aggregate marginal utilities.
        - max_exp_values_draws (np.ndarray): 1d array of shape
        (n_states, n_savings, n_income_shocks,) of the aggregate expected values.

    """
    max_value_per_state = jnp.take(
        value_state_choice_combs,
        reshape_state_choice_vec_to_mat,
        axis=0,
    ).max(axis=1)

    max_value_per_state_choice_comb = jnp.tensordot(
        sum_state_choice_to_state, max_value_per_state, axes=(0, 0)
    )

    exponential_value = jnp.exp(
        (value_state_choice_combs - max_value_per_state_choice_comb) / taste_shock_scale
    )
    sum_exponential_values_per_state = jnp.tensordot(
        sum_state_choice_to_state, exponential_value, axes=(1, 0)
    )

    numerator_choice_probs_times_marg_util = jnp.tensordot(
        sum_state_choice_to_state,
        jnp.multiply(exponential_value, marg_util_state_choice_combs),
        axes=(1, 0),
    )

    marginal_utils = jnp.divide(
        numerator_choice_probs_times_marg_util,
        sum_exponential_values_per_state,
    )

    log_sum = max_value_per_state + taste_shock_scale * jnp.log(
        sum_exponential_values_per_state
    )

    return (
        marginal_utils @ income_shock_weights,
        log_sum @ income_shock_weights,
    )
