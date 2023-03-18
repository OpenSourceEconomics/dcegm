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
    final_period_solution,  # noqa: U100
    choices_final,
    compute_next_period_wealth,
    compute_marginal_utility,
    taste_shock_scale,
    exogenous_savings_grid,
    income_shock_draws,
    income_shock_weights,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes solution to final period for policy and value function.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        states (np.ndarray): Collection of all possible states.
        savings_grid (np.ndarray): Array of shape (n_wealth_grid,) denoting the
            exogenous savings grid.
        options (dict): Options dictionary.
        compute_utility (callable): Function for computation of agent's utility.

    Returns:
        (tuple): Tuple containing

        - policy (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific policy function; of shape
            [n_states, n_discrete_choices, 2, 1.1 * (n_grid_wealth + 1)].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each state and each discrete choice.
        - value (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_states, n_discrete_choices, 2, 1.1 * (n_grid_wealth + 1)].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each state and each discrete choice.

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
