from typing import Tuple

import jax.numpy as jnp


def aggregate_marg_utils_exp_values(
    value_state_choice_specific: jnp.ndarray,
    marg_util_state_choice_specific: jnp.ndarray,
    reshape_state_choice_vec_to_mat: jnp.ndarray,
    transform_between_state_and_state_choice_vec: jnp.ndarray,
    taste_shock_scale: float,
    income_shock_weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the aggregate marginal utilities and expected values.

    Args:
        value_state_choice_choice_specific (jnp.ndarray): 3d array of shape
            (n_states * n_choices, n_exog_savings, n_income_shocks) of the value
            function for all state-choice combinations and income shocks.
        marg_util_state_choice_specific (jnp.ndarray): 3d array of shape
            (n_states * n_choices, n_exog_savings, n_income_shocks) of the marginal
            utility of consumption for all states-choice combinations and
            income shocks.
        resources_current_period (np.ndarray): 3d array of shape
            (n_state_choice_combs_current, n_exog_savings, n_stochastic_quad_points)
            containing the resources at the beginning of the current period.
        reshape_current_state_choice_vec_to_mat (np.ndarray): 2d array of shape
            (n_states_current, n_choices_current) that reshapes the current period
            vector of feasible state-choice combinations to a matrix of shape
            (n_choices, n_choices).
        transform_between_state_and_state_choice_vec (np.ndarray): 2d boolean
            array of shape (n_states_current, n_feasible_state_choice_combs_current)
            indicating which state vector belongs to which state-choice combination in
            the current period.
            (i) contract state-choice level arrays to the state level by summing
                over state-choice combinations.
            (ii) to expand state level arrays to the state-choice level.
        taste_shock_scale (float): The taste shock scale.
        income_shock_weights (jnp.ndarray): 1d array of shape
            (n_stochastic_quad_points,) containing the weights of the income shock
            quadrature.

    Returns:
        tuple:

        - marg_util (np.ndarray): 2d array of shape (n_states, n_exog_savings)
            of the state-specific aggregate marginal utilities.
        - expected_value (np.ndarray): 2d array of shape (n_states, n_exog_savings)
            of the state-specific aggregate expected values.

    """
    max_value_per_state = jnp.take(
        value_state_choice_specific,
        reshape_state_choice_vec_to_mat,
        axis=0,
    ).max(axis=1)

    max_value_per_state_choice_comb = jnp.tensordot(
        transform_between_state_and_state_choice_vec, max_value_per_state, axes=(0, 0)
    )

    value_exponential = jnp.exp(
        (value_state_choice_specific - max_value_per_state_choice_comb)
        / taste_shock_scale
    )
    sum_value_exponential_per_state = jnp.tensordot(
        transform_between_state_and_state_choice_vec, value_exponential, axes=(1, 0)
    )

    product_choice_probs_and_marg_util = jnp.tensordot(
        transform_between_state_and_state_choice_vec,
        jnp.multiply(value_exponential, marg_util_state_choice_specific),
        axes=(1, 0),
    )
    marg_util = jnp.divide(
        product_choice_probs_and_marg_util,
        sum_value_exponential_per_state,
    )

    log_sum = max_value_per_state + taste_shock_scale * jnp.log(
        sum_value_exponential_per_state
    )

    return (
        marg_util @ income_shock_weights,
        log_sum @ income_shock_weights,
    )
