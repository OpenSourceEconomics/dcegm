from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.fast_upper_envelope import fast_upper_envelope_wrapper


def convert_params_to_dict(params: pd.DataFrame) -> Dict[str, float]:
    """Transforms params DataFrame into a dictionary.

    Checks if given params DataFrame contains taste shock scale, interest rate
    and discount factor.

    Args:
        params (pd.DataFrame): Params DataFrame.

    Returns:
        dict: Dictionary with index "category" dropped and column
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
) -> Tuple[Callable, Callable, Callable, Callable, Callable, Callable, Callable]:
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
        tuple:

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
        - compute_upper_envelope (Callable): Function for calculating the upper envelope
            of the policy and value function. If the number of discrete choices is 1,
            this function is a dummy function that returns the policy and value
            function as is, without performing a fast upper envelope scan.
        - transition_function (Callable): Partialled transition function that returns
            transition probabilities for each state.

    """
    compute_utility = partial(
        user_utility_functions["utility"],
        params_dict=params_dict,
    )

    compute_marginal_utility = partial(
        user_utility_functions["marginal_utility"],
        params_dict=params_dict,
    )

    compute_inverse_marginal_utility = partial(
        user_utility_functions["inverse_marginal_utility"],
        params_dict=params_dict,
    )

    compute_value = partial(
        calc_current_value,
        discount_factor=params_dict["beta"],
        compute_utility=compute_utility,
    )

    compute_next_period_wealth = partial(
        user_budget_constraint,
        params_dict=params_dict,
        options=options,
    )

    transition_function = partial(
        exogenous_transition_function, params_dict=params_dict
    )

    if options["n_discrete_choices"] == 1:
        compute_upper_envelope = _return_policy_and_value
    else:
        compute_upper_envelope = fast_upper_envelope_wrapper

    return (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_value,
        compute_next_period_wealth,
        compute_upper_envelope,
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
        np.ndarray: The current value.

    """
    utility = compute_utility(consumption, choice)
    value = utility + discount_factor * next_period_value

    return value


def create_multi_dim_arrays(
    state_choice_space: np.ndarray,
    options: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create multi-dimensional arrays for endogenous grid, policy and value function.

    We include one additional grid point (n_grid_wealth + 1) to M,
    since we want to set the first position (j=0) to M_t = 0 for all time
    periods.

    Moreover, we add 10% extra space filled with nans, since, in the upper
    envelope step, the endogenous wealth grid might be augmented to the left
    in order to accurately describe potential non-monotonicities (and hence
    discontinuities) near the start of the grid.


    Note that, in the Upper Envelope step, we drop suboptimal points from the original
    grid and add new ones (kink points as well as the corresponding interpolated values
    of the policy and value functions).

    Args:
        state_space (np.ndarray): Collection of all possible states.
        options (dict): Options dictionary.

    Returns:
        tuple:

        - endog_grid_container (np.ndarray): "Empty" 3d np.ndarray storing the
            endogenous grid for each state and each discrete choice.
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
        - policy_container (np.ndarray): "Empty" 3d np.ndarray storing the
            choice-specific policy function for each state and each discrete choice
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
        - value_container (np.ndarray): "Empty" 3d np.ndarray storing the
            choice-specific value functions for each state and each discrete choice.
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].

    """
    n_grid_wealth = options["grid_points_wealth"]
    n_state_choicess = state_choice_space.shape[0]

    endog_grid_container = np.empty((n_state_choicess, int(1.1 * n_grid_wealth)))
    policy_container = np.empty((n_state_choicess, int(1.1 * n_grid_wealth)))
    value_container = np.empty((n_state_choicess, int(1.1 * n_grid_wealth)))
    endog_grid_container[:] = np.nan
    policy_container[:] = np.nan
    value_container[:] = np.nan

    return endog_grid_container, policy_container, value_container


def _return_policy_and_value(
    endog_grid, policy, value, expected_value_zero_savings, **kwargs
):
    endog_grid = np.append(0, endog_grid)
    policy = np.append(0, policy)
    value = np.append(expected_value_zero_savings, value)
    return endog_grid, policy, value
