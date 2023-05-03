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
    final_period_states: np.ndarray,
    options: Dict[str, int],
    compute_utility: Callable,
    final_period_solution: Callable,
    choices_final: np.ndarray,
    compute_next_period_wealth: Callable,
    compute_marginal_utility: Callable,
    taste_shock: float,
    exogenous_savings_grid: np.ndarray,
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
        compute_next_period_wealth (callable): User-defined function to compute the
            agent's wealth of the next period (t + 1). The inputs
            ```saving```, ```shock```, ```params``` and ```options```
            are already partialled in.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        taste_shock_scale (float): The taste shock scale.
        exogenous_savings_grid (np.ndarray): Array of shape (n_wealth_grid,) denoting
            the exogenous savings grid.
        income_shock_draws (np.ndarray): 1d array of shape (n_quad_points,) containing
            the Hermite quadrature points.
        income_shock_weights (np.ndarrray): Weights for each stoachstic shock draw.
            Shape is (n_stochastic_quad_points)

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
    n_choices = options["n_discrete_choices"]

    # Compute beginning of period wealth in last period
    resources_last_period = vmap(
        vmap(
            vmap(compute_next_period_wealth, in_axes=(None, None, 0)),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, None, None),
    )(final_period_states, exogenous_savings_grid, income_shock_draws)

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
                vmap(
                    final_period_solution_partial,
                    in_axes=(None, 0, None),
                ),
                in_axes=(None, 0, None),
            ),
            in_axes=(None, None, 0),
        ),
        in_axes=(0, 0, None),
    )(final_period_states, resources_last_period, np.arange(n_choices, dtype=int))

    partial_aggregate = partial(
        aggregate_marg_utilites_and_values_over_choices,
        taste_shock=taste_shock,
    )

    # Weight all draws and aggregate over choices
    marginal_utils_draws, max_exp_values_draws = vmap(
        vmap(
            vmap(partial_aggregate, in_axes=(1, 1, None, 0)),
            in_axes=(1, 1, None, None),
        ),
        in_axes=(0, 0, 0, None),
    )(
        final_value,
        marginal_utilities_choices,
        choices_final,
        income_shock_weights,
    )

    # Aggregate the weighted arrays
    marginal_utils = marginal_utils_draws.sum(axis=2)
    max_exp_values = max_exp_values_draws.sum(axis=2)

    # Choose which draw we take for policy and value function as those are note saved
    # with respect to the draws
    middle_of_draws = int(income_shock_draws.shape[0] + 1 / 2)

    # breakpoint()

    return (
        np.repeat(
            resources_last_period[:, np.newaxis, :, middle_of_draws], n_choices, axis=1
        ),
        final_policy[..., middle_of_draws],
        final_value[..., middle_of_draws],
        marginal_utils,
        max_exp_values,
    )
