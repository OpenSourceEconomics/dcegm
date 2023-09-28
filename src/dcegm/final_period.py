from typing import Callable
from typing import Dict
from typing import Tuple

import jax.numpy as jnp
from jax import vmap

"""Wrapper function to solve the final period of the model."""


def solve_final_period(
    state_objects_final_period: jnp.ndarray,
    compute_final_period: Callable,
    resources_beginning_of_period: Dict[str, float],
    params: Dict[str, float],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes solution to final period for policy and value function.
    In the last period, everything is consumed, i.e. consumption = savings.
    Args:
        final_period_choice_states (np.ndarray): Collection of all possible
              state-choice combinations in the final period.
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

    resources_final_period = resources_beginning_of_period[
        state_objects_final_period["idx_parent_states"]
    ]

    marg_util_interpolated, value_interpolated, policy_final = vmap(
        vmap(
            vmap(
                compute_final_period,
                in_axes=(None, 0, None),
            ),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, 0, None),
    )(
        state_objects_final_period["state_choice_mat"],
        resources_final_period,
        params,
    )

    # Choose which draw we take for policy and value function as those are not
    # saved with respect to the draws
    middle_of_draws = int(value_interpolated.shape[2] + 1 / 2)

    results = {}
    results["value"] = value_interpolated[:, :, middle_of_draws]
    results["policy_left"] = policy_final[:, :, middle_of_draws]
    results["policy_right"] = policy_final[:, :, middle_of_draws]
    results["endog_grid"] = resources_final_period[:, :, middle_of_draws]

    return results, marg_util_interpolated, value_interpolated, policy_final
