"""Wrapper function to solve the final period of the model."""
from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
from dcegm.marg_utilities_and_exp_value import (
    aggregate_marg_utilites_and_values_over_choices,
)
from jax import vmap


def final_period_wrapper(
    final_period_choice_states: np.ndarray,
    options: Dict[str, int],
    compute_utility: Callable,
    final_period_solution: Callable,
    sum_state_choices_to_state,
    resources_last_period: np.ndarray,
    compute_marginal_utility: Callable,
    taste_shock_scale: float,
    income_shock_draws: np.ndarray,
    income_shock_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes solution to final period for policy and value function.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        final_period_states (np.ndarray): Collection of all possible states.
        options (dict): Options dictionary.
        compute_utility (callable): Function for computation of agent's utility.
        final_period_solution (callable): User-supplied function for solving the agent's
            last period.
        choices_final (np.ndarray): binary array indicating if choice is possible in
            final states.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        taste_shock_scale (float): The taste shock scale.
        income_shock_draws (np.ndarray): 1d array of shape (n_quad_points,) containing
            the Hermite quadrature points.
        income_shock_weights (np.ndarrray): 1d array of shape (n_stochastic_quad_points)
            with weights for each stoachstic shock draw.

    Returns:
        tuple:

        - endog_grid_final (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the
            endogenous wealth grid for all final states, choices, and
            end of period assets from the period before.
        - final_policy (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the optimal
            policy for all final states, choices, end of period assets, and
            income shocks.
        - final_value (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, choices, end of period assets, and
            income shocks.
        - marginal_utilities_choices (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, choices, end of period assets, and
            income shocks.
        - max_exp_values (np.ndarray): 4d array of shape
            (n_states, n_choices, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, choices, end of period assets, and
            income shocks.

    """
    final_period_solution_partial = partial(
        final_period_solution,
        options=options,
        params_dict={},
        compute_utility=compute_utility,
        compute_marginal_utility=compute_marginal_utility,
    )

    # Compute for each wealth grid point the optimal policy and value function as well
    # as the marginal utility of consumption for all choices.
    final_policy, final_value, marginal_utilities_choices = vmap(
        vmap(
            vmap(
                final_period_solution_partial,
                in_axes=(None, 0, None),
            ),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, 0, 0),
    )(
        final_period_choice_states[:, :-1],
        resources_last_period,
        final_period_choice_states[:, -1],
    )

    # Aggregate the marginal utilities and expected values over all choices
    marginal_utils_draws, max_exp_values_draws = aggregate_marg_utils_exp_values(
        final_value_state_choice=final_value,
        marg_util_state_choice=marginal_utilities_choices,
        sum_state_choices_to_state=sum_state_choices_to_state,
        taste_shock_scale=taste_shock_scale,
    )

    # Aggregate the weighted arrays
    marginal_utils = marginal_utils_draws @ income_shock_weights
    max_exp_values = max_exp_values_draws @ income_shock_weights

    # Choose which draw we take for policy and value function as those are note saved
    # with respect to the draws
    middle_of_draws = int(income_shock_draws.shape[0] + 1 / 2)

    return (
        final_policy[:, :, middle_of_draws],
        final_value[:, :, middle_of_draws],
        marginal_utils,
        max_exp_values,
    )


def aggregate_marg_utils_exp_values(
    final_value_state_choice: np.ndarray,
    marg_util_state_choice: np.ndarray,
    sum_state_choices_to_state: np.ndarray,
    taste_shock_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the aggregate marginal utilities and expected values.

    Args:
        final_value_state_choice (np.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the value function for all
            states, choices, and income shocks.
        marg_util_state_choice (np.ndarray): 2d array of shape
            (n_states * n_choices, n_income_shocks) of the marginal utility of
            consumption for all states, choices, and income shocks.
        sum_state_choices_to_state (np.ndarray): 2d array of shape
            (n_states, n_states * n_choices) with state_space size  with ones, where
            state-choice belongs to state.
        taste_shock_scale (float): The taste shock scale.

    Returns:
        tuple:

        - marginal_utils_draws (np.ndarray): 1d array of shape
        (n_states, n_savings, n_income_shocks,) of the aggregate marginal utilities.
        - max_exp_values_draws (np.ndarray): 1d array of shape
        (n_states, n_savings, n_income_shocks,) of the aggregate expected values.

    """

    rescale_value = np.amax(final_value_state_choice)
    exp_value = np.exp(final_value_state_choice - rescale_value)
    sum_exp = np.tensordot(sum_state_choices_to_state, exp_value, axes=(1, 0))

    max_exp_values_draws = rescale_value + taste_shock_scale * np.log(sum_exp)
    marginal_utils_draws = np.divide(
        np.tensordot(
            sum_state_choices_to_state,
            np.multiply(exp_value, marg_util_state_choice),
            axes=(1, 0),
        ),
        sum_exp,
    )
    return marginal_utils_draws, max_exp_values_draws
