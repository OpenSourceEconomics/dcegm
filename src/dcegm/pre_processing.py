from functools import partial
from typing import Dict
from typing import Tuple

import numpy as np
from dcegm.aggregate_policy_value import calc_current_period_value


def get_partial_functions(
    params,
    options,
    quad_points,
    quad_weights,
    exogenous_savings_grid,
    user_utility_func,
    user_marginal_utility_func,
    user_inverse_marginal_utility_func,
    user_budget_constraint,
    user_marginal_next_period_wealth,
):

    compute_utility = partial(
        user_utility_func,
        params=params,
    )
    compute_marginal_utility = partial(
        user_marginal_utility_func,
        params=params,
    )
    compute_inverse_marginal_utility = partial(
        user_inverse_marginal_utility_func,
        params=params,
    )
    compute_current_value = partial(
        calc_current_period_value,
        discount_factor=params.loc[("beta", "beta"), "value"],
        compute_utility=compute_utility,
    )
    compute_next_wealth_matrices = partial(
        user_budget_constraint,
        savings_grid=exogenous_savings_grid,
        income_shock=quad_points,
        params=params,
        options=options,
    )
    compute_next_marginal_wealth = partial(
        user_marginal_next_period_wealth,
        params=params,
        options=options,
    )
    return (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_current_value,
        compute_next_wealth_matrices,
        compute_next_marginal_wealth,
    )


def create_multi_dim_arrays(
    state_space: np.ndarray,
    options: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create multi-diminesional array for storing the policy and value function.

    Note that we add 10% extra space filled with nans, since, in the upper
    envelope step, the endogenous wealth grid might be augmented to the left
    in order to accurately describe potential non-monotonicities (and hence
    discontinuities) near the start of the grid.

    We include one additional grid point (n_grid_wealth + 1) to M,
    since we want to set the first position (j=0) to M_t = 0 for all time
    periods.

    Moreover, the lists have variable length, because the Upper Envelope step
    drops suboptimal points from the original grid and adds new ones (kink
    points as well as the corresponding interpolated values of the consumption
    and value functions).

    Args:
        options (dict): Options dictionary.
        state_space (np.ndarray): Collection of all possible states.


    Returns:
        (tuple): Tuple containing:

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
    n_grid_wealth = options["grid_points_wealth"]
    n_choices = options["n_discrete_choices"]
    n_states = state_space.shape[0]

    policy_arr = np.empty((n_states, n_choices, 2, int(1.1 * n_grid_wealth + 1)))
    value_arr = np.empty((n_states, n_choices, 2, int(1.1 * n_grid_wealth + 1)))
    policy_arr[:] = np.nan
    value_arr[:] = np.nan

    return policy_arr, value_arr
