from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.aggregate_policy_value import calc_value


def get_partial_functions(
    params: pd.DataFrame,
    options: Dict[str, int],
    user_utility_functions: Dict[str, Callable],
    user_budget_constraint: Callable,
    exogenous_transition_function: Callable,
) -> Tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """Create partial functions from user supplied functions. This is the function
    where we handle all custom provided parameters.
    Args:
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        user_utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of:

            (i) utility
            (ii) inverse marginal utility
            (iii) next period marginal utility
        user_budget_constraint (callable): Callable budget constraint.
        exogenous_transition_function (callable): User-supplied function returning for
            each state a transition matrix vector.
    Returns:
        - compute_utility (callable): Function for computation of agent's utility.
        - compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        - compute_inverse_marginal_utility (Callable): Function for calculating the
            inverse marginal utiFality, which takes the marginal utility as only input.
        - compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
        - compute_next_wealth_matrices (callable): User-defined function to compute the
            agent's wealth matrices of the next period (t + 1). The inputs
            ```savings_grid```, ```income_shocks```, ```params``` and ```options```
            are already partialled in.
        - transition_function (Callable): Partialled transition function return
            transition vector for each state.
    """
    compute_utility = partial(
        user_utility_functions["utility"],
        params=params,
    )
    compute_marginal_utility = partial(
        user_utility_functions["marginal_utility"],
        params=params,
    )
    compute_inverse_marginal_utility = partial(
        user_utility_functions["inverse_marginal_utility"],
        params=params,
    )
    compute_value = partial(
        calc_value,
        discount_factor=params.loc[("beta", "beta"), "value"],
        compute_utility=compute_utility,
    )
    compute_next_wealth_matrices = partial(
        user_budget_constraint,
        params=params,
        options=options,
    )
    transition_function = partial(exogenous_transition_function, params=params)

    return (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_value,
        compute_next_wealth_matrices,
        transition_function,
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
