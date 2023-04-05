from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
from dcegm.marg_utilities_and_exp_value import (
    aggregate_marg_utilites_and_values_over_choices_and_weight,
)
from jax import vmap


def final_period_wrapper(
    final_period_states: np.ndarray,
    options: Dict[str, int],
    compute_utility: Callable,
    final_period_solution: Callable,  # noqa: U100
    choices_final: np.ndarray,
    compute_next_period_wealth: Callable,
    compute_marginal_utility: Callable,
    taste_shock_scale: float,
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
        (tuple): Tuple containing

        - endog_grid (np.ndarray): The endogenous wealth grid for all final states
            and end of period assets from the period before.
        - policy (np.ndarray): The optimal policy defined on the endogenous wealth grid.
        - value (np.ndarray): The value function defined on the endogenous wealth grid.
        - marginal_utilities_choices (np.ndarray): The marginal utility of consumption
            for all final states, end of period asset and income wealth grid.
        - max_exp_values (np.ndarray): The maximum expected value of the value function
            for all final states, end of period asset and income wealth grid.

    """
    n_choices = options["n_discrete_choices"]

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
        aggregate_marg_utilites_and_values_over_choices_and_weight,
        taste_shock_scale=taste_shock_scale,
    )

    marginal_utils_draws, max_exp_values_draws = vmap(
        vmap(
            vmap(partial_aggregate, in_axes=(None, 1, 1, 0)), in_axes=(None, 1, 1, None)
        ),
        in_axes=(0, 0, 0, None),
    )(
        choices_final,
        marginal_utilities_choices,
        final_value,
        income_shock_weights,
    )

    marginal_utils = marginal_utils_draws.sum(axis=2)
    max_exp_values = max_exp_values_draws.sum(axis=2)
    middle_of_draws = int(income_shock_draws.shape[0] + 1 / 2)

    return (
        resources_last_period[:, :, middle_of_draws],
        final_policy[:, :, :, middle_of_draws],
        final_value[:, :, :, middle_of_draws],
        marginal_utils,
        max_exp_values,
    )
