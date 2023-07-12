from typing import Tuple

import jax.numpy as jnp
from jax import numpy as jnp


def aggregate_marg_utils_exp_values(
    value_state_choice_combs: jnp.ndarray,
    marg_util_state_choice_combs: jnp.ndarray,
    reshape_state_choice_vec_to_mat: jnp.ndarray,
    transform_between_state_and_state_choice_vec: jnp.ndarray,
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
        transform_between_state_and_state_choice_vec, max_value_per_state, axes=(0, 0)
    )

    value_exp = jnp.exp(
        (value_state_choice_combs - max_value_per_state_choice_comb) / taste_shock_scale
    )
    sum_value_exp_per_state = jnp.tensordot(
        transform_between_state_and_state_choice_vec, value_exp, axes=(1, 0)
    )

    product_choice_probs_and_marg_util = jnp.tensordot(
        transform_between_state_and_state_choice_vec,
        jnp.multiply(value_exp, marg_util_state_choice_combs),
        axes=(1, 0),
    )

    marg_util = jnp.divide(
        product_choice_probs_and_marg_util,
        sum_value_exp_per_state,
    )

    log_sum = max_value_per_state + taste_shock_scale * jnp.log(sum_value_exp_per_state)

    return (
        marg_util @ income_shock_weights,
        log_sum @ income_shock_weights,
    )
