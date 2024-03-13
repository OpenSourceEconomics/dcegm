"""Wrapper to solve the final period of the model."""
from typing import Callable
from typing import Dict
from typing import Tuple

import jax.numpy as jnp
from jax import vmap


def solve_final_period(
    idx_parent_states,
    state_choice_mat,
    resources_beginning_of_period: jnp.ndarray,
    params: Dict[str, float],
    compute_utility: Callable,
    compute_marginal_utility: Callable,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Computes solution to final period for policy and value function.
    In the last period, everything is consumed, i.e. consumption = savings.
    Args:


    Returns:
        tuple:
        - marginal_utilities_choices (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the marginal utility of
            consumption for all final states, end of period assets, and
            income shocks.
        - final_value (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, end of period assets, and
            income shocks.
        - final_policy (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the optimal
            policy for all final states, end of period assets, and
            income shocks.
    """

    resources = resources_beginning_of_period[idx_parent_states]

    value, marg_util = vmap(
        vmap(
            vmap(
                calculate_value_and_marg_util_for_each_gridpoint,
                in_axes=(None, 0, None, None, None),
            ),
            in_axes=(None, 0, None, None, None),
        ),
        in_axes=(0, 0, None, None, None),
    )(
        state_choice_mat,
        resources,
        params,
        compute_utility,
        compute_marginal_utility,
    )

    # Choose which draw we take for policy and value function as those are not
    # saved with respect to the draws
    middle_of_draws = int(value.shape[2] + 1 / 2)

    value_calc = value[:, :, middle_of_draws]
    # The policy in the last period is eat it all. Either as bequest or by consuming.
    # The user defines this by the bequest functions.
    resources_to_save = resources[:, :, middle_of_draws]
    nans_to_add = jnp.full(
        (resources_to_save.shape[0], int(resources_to_save.shape[1] * 0.2)), jnp.nan
    )
    policy_left = policy_right = endog_grid = jnp.append(
        resources_to_save,
        nans_to_add,
        axis=1,
    )
    value_final = jnp.append(value_calc, nans_to_add, axis=1)

    return (
        value_final,
        policy_left,
        policy_right,
        endog_grid,
        value,
        marg_util,
    )


def calculate_value_and_marg_util_for_each_gridpoint(
    state_choice_vec, resources, params, compute_utility, compute_marginal_utility
):
    value = compute_utility(
        **state_choice_vec,
        resources=resources,
        params=params,
    )

    marg_util = compute_marginal_utility(
        **state_choice_vec,
        resources=resources,
        params=params,
    )

    return value, marg_util
