from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import jax
import numpy as np
import pandas as pd


def params_todict(params):
    """Transforms params DataFrame into a dictionary. Checks if given params DataFrame
    contains taste shock scale, interest rate and discount factor.

    Args:
        params (pd.DataFrame): Params DataFrame.
    Returns:
        params_dict (dict): Params Data Frame without index "category" and column
        "comment" transformed into dictionary.

    """

    keys = params.index.droplevel("category").tolist()
    values = params["value"].tolist()
    params_dict = dict(zip(keys, values))

    if "interest_rate" not in params_dict:  # interest rate
        raise ValueError("Interest rate must be provided in params.")
    if "lambda" not in params_dict:  # taste shock scale
        raise ValueError("Taste shock scale must be provided in params.")
    if "beta" not in params_dict:  # discount factor
        raise ValueError("Discount factor must be provided in params.")

    return params_dict


def get_partial_functions(
    params_dict: dict,
    options: Dict[str, int],
    user_utility_functions: Dict[str, Callable],
    user_budget_constraint: Callable,
    exogenous_transition_function: Callable,
) -> Tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """Create partial functions from user supplied functions.

    Args:
        params_dict (dict): Dictionary containing model parameters.
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
            inverse marginal utility, which takes the marginal utility as only input.
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
        params_dict=params_dict,
    )
    compute_marginal_utility = jax.jit(
        partial(
            user_utility_functions["marginal_utility"],
            params_dict=params_dict,
        )
    )

    compute_inverse_marginal_utility = jax.jit(
        partial(
            user_utility_functions["inverse_marginal_utility"],
            params_dict=params_dict,
        )
    )

    compute_value = jax.jit(
        partial(
            calc_current_value,
            discount_factor=params_dict["beta"],
            compute_utility=compute_utility,
        )
    )

    compute_next_period_wealth = jax.jit(
        partial(
            user_budget_constraint,
            params_dict=params_dict,
            options=options,
        )
    )
    transition_function = partial(
        exogenous_transition_function, params_dict=params_dict
    )
    return (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_value,
        compute_next_period_wealth,
        transition_function,
    )


def calc_current_value(
    consumption: np.ndarray,
    next_period_value: np.ndarray,
    choice: int,
    discount_factor: float,
    compute_utility: Callable,
) -> np.ndarray:
    """Compute the agent's current value.

    We only support the standard value function, where the current utility and
    the discounted next period value have a sum format.

    Args:
        consumption (np.ndarray): Level of the agent's consumption.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        next_period_value (np.ndarray): The value in the next period.
        choice (int): The current discrete choice.
        compute_utility (callable): User-defined function to compute the agent's
            utility. The input ``params``` is already partialled in.
        discount_factor (float): The discount factor.

    Returns:
        - value (np.ndarray): The current value.

    """
    utility = compute_utility(consumption, choice)
    value = utility + discount_factor * next_period_value

    return value


def create_multi_dim_arrays(
    state_space: np.ndarray,
    options: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create multi-dimensional array for storing the policy and value function.

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


def get_possible_choices_array(
    state_space: np.ndarray,
    state_indexer: np.ndarray,
    get_state_specific_choice_set: Callable,
    options: Dict[str, int],
) -> np.ndarray:
    """Create binary array for storing the possible choices for each state.

    Args:
        state_space (np.ndarray): Collection of all possible states.
        state_indexer (np.ndarray): 2d array of shape (n_periods, n_choices) containing
            the indexer object that maps states to indices in the state space.
        get_state_specific_choice_set (Callable): User-supplied function returning for
            each state all possible choices.
        options (dict): Options dictionary.

    Returns:
        choices_array (np.ndarray): binary array storing the possible choices for
        each state. If choices_array[state_index, choice] = 1, then the choice
        is contained in the set of possible choices of the state. If
        choices_array[state_index, choice] = 0, then the choice is not contained
        in the set of possible choices of the state.

    """
    n_choices = options["n_discrete_choices"]
    choices_array = np.zeros((state_space.shape[0], n_choices))
    for index in range(state_space.shape[0]):
        state = state_space[index]
        choice_set = get_state_specific_choice_set(state, state_space, state_indexer)
        choices_array[index, choice_set] = 1

    return choices_array
