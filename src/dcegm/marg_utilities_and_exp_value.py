from typing import Callable
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import numpy as jnp
from jax import vmap


# def marginal_util_and_exp_max_value_states_period(
#     endog_grid_child_states: np.ndarray,
#     policy_child_states: np.ndarray,
#     value_child_states: np.ndarray,
#     resources_next_period: np.ndarray,
#     choices_child_states: np.ndarray,
#     income_shock_weights: np.ndarray,
#     taste_shock_scale: float,
#     compute_marginal_utility: Callable,
#     compute_value: Callable,
# ):
#     """Compute the child-state specific marginal utility and expected maximum value.
#
#     The underlying algorithm is the Endogenous-Grid-Method (EGM).
#
#     Args:
#         endog_grid_child_states (np.ndarray): 3d array containing the endogenous
#             wealth grid of the child_states. Shape (n_child_states_period, n_choice,
#             n_grid_wealth).
#         policy_child_states (np.ndarray): 3d array containing the corresponding
#             policy function values of the endogenous wealth grid of the child_states
#             shape (n_child_states_period, n_choice, n_grid_wealth).
#         value_child_states (np.ndarray): 3d array containing the corresponding
#             value function values of the endogenous wealth grid of the child_states
#             shape (n_child_states_period, n_choice, n_grid_wealth).
#         exogenous_savings_grid (np.array): Exogenous savings grid.
#         state_space_next (np.ndarray): 2d array of shape
#             (n_child_states, n_state_variables + 1) containing the possible child_states
#             in the next period (t + 1) that can be reached from the current period.
#         choices_child_states (np.ndarray): 2d boolean array of shape
#             (n_states_period, n_choices) denoting if the state-choice combinations are
#             feasible in the given period.
#         income_shock_draws (np.ndarray): 1d array of shape (n_quad_points,)
#             containing the Hermite quadrature points.
#         income_shock_weights (np.ndarrray): 1d array of shape
#             (n_stochastic_quad_points) with weights for each stoachstic shock draw.
#         taste_shock_scale (float): The taste shock scale parameter.
#         compute_next_period_wealth (callable): User-defined function to compute the
#             agent's wealth  of the next period (t + 1). The inputs
#             ```saving```, ```income_shock```, ```params``` and ```options```
#             are already partialled in.
#         compute_marginal_utility (callable): User-defined function to compute the
#             agent's marginal utility. The input ```params``` is already partialled in.
#         compute_value (callable): User-defined function to compute
#             the agent's value function in the credit-constrained area. The inputs
#             ```params``` and ```compute_utility``` are already partialled in.
#
#     Returns:
#         tuple:
#
#         - marg_utils (jnp.ndarray): 2d array of shape (n_states_period, n_grid_wealth).
#             containing the child-state specific marginal utilities,
#             weighted by the vector of income shocks.
#         - emax (jnp.ndarray): 2d array of shape (n_states_period, n_grid_wealth)
#             containing the child-state specific expected maximum values,
#             weighted by the vector of income shocks.
#
#     """
#     marg_util_pre_lim, values_pre_lim = vmap(
#         vectorized_marginal_util_and_exp_max_value,
#         in_axes=(0, 0, 0, 0, 0, None, None, None, None),
#     )(
#         endog_grid_child_states,
#         resources_next_period,
#         policy_child_states,
#         value_child_states,
#         choices_child_states,
#         income_shock_weights,
#         taste_shock_scale,
#         compute_marginal_utility,
#         compute_value,
#     )
#
#     return (
#         marg_util_pre_lim,
#         values_pre_lim,
#     )


# def vectorized_marginal_util_and_exp_max_value(
#     endog_grid_child_state: jnp.ndarray,
#     next_period_wealth: jnp.ndarray,
#     choice_policies_child_state: jnp.ndarray,
#     choice_values_child_state: jnp.ndarray,
#     choice_set_indices: jnp.ndarray,
#     income_shock_weight: jnp.ndarray,
#     taste_shock_scale: float,
#     compute_marginal_utility: Callable,
#     compute_value: Callable,
# ) -> Tuple[float, float]:
#     """Compute the child-state specific marginal utility and expected maximum value.
#
#     The underlying algorithm is the Endogenous-Grid-Method (EGM).
#
#     Args:
#         endog_grid_child_state (jnp.ndarray): 2d array containing the endogenous
#             wealth grid of each child state. Shape (n_choice, n_grid_wealth).
#         next_period_wealth (jnp.ndarray): 1d array of the agent's wealth of the next
#             period (t + 1) for each exogenous savings and income shock grid point.
#             Shape (n_grid_wealth, n_income_shocks).
#         choice_policies_child_state (jnp.ndarray): 2d array containing the corresponding
#             policy function values of the endogenous wealth grid of each child state.
#             Shape (n_choice, n_grid_wealth).
#         choice_values_child_state (jnp.ndarray): 2d array containing the corresponding
#             value function values of the endogenous wealth grid of each child state.
#             Shape (n_choice, n_grid_wealth).
#         choice_set_indices (jnp.ndarray): The agent's (restricted) choice set in the
#             given state of shape (n_choices,).
#         income_shock_weight (jnp.ndarray): Weights of stochastic shock draws.
#         taste_shock_scale (float): The taste shock scale parameter.
#         compute_marginal_utility (callable): User-defined function to compute the
#             agent's marginal utility. The input ```params``` is already partialled in.
#         compute_value (callable): User-defined function to compute
#             the agent's value function in the credit-constrained area. The inputs
#             ```params``` and ```compute_utility``` are already partialled in.
#
#     Returns:
#         tuple:
#
#         - marg_utils_weighted (jnp.ndarray): 2d array of shape
#             (n_child_states, n_grid_wealth) containing the child-state specific
#             marginal utilities, weighted by the vector of income shocks.
#         - emax_weighted (jnp.ndarray): 2d array of shape (n_child_states, n_grid_wealth)
#             containing the child-state specific expected maximum values,
#             weighted by the vector of income shocks.
#
#     """
#
#     # Interpolate the optimal consumption choice on the wealth grid and calculate the
#     # corresponding marginal utilities.
#     marg_utils, values = get_values_and_marginal_utilities(
#         compute_marginal_utility=compute_marginal_utility,
#         compute_value=compute_value,
#         next_period_wealth=next_period_wealth,
#         endog_grid_child_state=endog_grid_child_state,
#         choice_policies_child_state=choice_policies_child_state,
#         choice_values_child_state=choice_values_child_state,
#     )
#
#     return marg_utils, values


def aggregate_marg_utils_exp_values(
    final_value_state_choice: jnp.ndarray,
    state_times_state_choice_mat: jnp.ndarray,
    marg_util_state_choice: jnp.ndarray,
    sum_state_choices_to_state: jnp.ndarray,
    taste_shock_scale: float,
    income_shock_weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the aggregate marginal utilities and expected values.

    Args:
        final_value_state_choice (jnp.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the value function for all
            states, choices, and income shocks.
        marg_util_state_choice (jnp.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the marginal utility of
            consumption for all states, choices, and income shocks.
        sum_state_choices_to_state (jnp.ndarray): 2d array of shape
            (n_states, n_states * n_choices) with state_space size  with ones, where
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
        final_value_state_choice,
        state_times_state_choice_mat,
        axis=0,
    ).max(axis=1)
    rescale_value = jnp.tensordot(
        sum_state_choices_to_state, max_value_per_state, axes=(0, 0)
    )

    exp_value = jnp.exp((final_value_state_choice - rescale_value)) ** (
        1 / taste_shock_scale
    )
    sum_exp = jnp.tensordot(sum_state_choices_to_state, exp_value, axes=(1, 0))

    max_exp_values_draws = max_value_per_state + taste_shock_scale * jnp.log(sum_exp)
    marginal_utils_draws = jnp.divide(
        jnp.tensordot(
            sum_state_choices_to_state,
            jnp.multiply(exp_value, marg_util_state_choice),
            axes=(1, 0),
        ),
        sum_exp,
    )
    return (
        marginal_utils_draws @ income_shock_weights,
        max_exp_values_draws @ income_shock_weights,
    )
