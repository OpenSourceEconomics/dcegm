from typing import Tuple

import jax.numpy as jnp
from jax import numpy as jnp


def aggregate_marg_utils_exp_values(
    value_state_choices: jnp.ndarray,
    marg_util_state_choices: jnp.ndarray,
    map_state_to_state_choices: jnp.ndarray,
    sum_state_choices_to_state: jnp.ndarray,
    taste_shock_scale: float,
    income_shock_weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the aggregate marginal utilities and expected values.

    Args:
        value_state_choices (jnp.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the value function for all
            states, choices, and income shocks.
        marg_util_state_choices (jnp.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the marginal utility of
            consumption for all states, choices, and income shocks.
        sum_state_choices_to_state (jnp.ndarray): 2d array of shape
            (n_states, n_state_choices) with state_space size  with ones, where
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
        value_state_choices,
        map_state_to_state_choices,
        axis=0,
    ).max(axis=1)
    rescale_value = jnp.tensordot(
        sum_state_choices_to_state, max_value_per_state, axes=(0, 0)
    )

    exp_value = jnp.exp((value_state_choices - rescale_value) / taste_shock_scale)
    sum_exp = jnp.tensordot(sum_state_choices_to_state, exp_value, axes=(1, 0))

    max_exp_values_draws = max_value_per_state + taste_shock_scale * jnp.log(sum_exp)
    marginal_utils_draws = jnp.divide(
        jnp.tensordot(
            sum_state_choices_to_state,
            jnp.multiply(exp_value, marg_util_state_choices),
            axes=(1, 0),
        ),
        sum_exp,
    )

    return (
        marginal_utils_draws @ income_shock_weights,
        max_exp_values_draws @ income_shock_weights,
    )
