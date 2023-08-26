from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import jax.numpy as jnp
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
       t    (i) utility
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
    compute_utility = user_utility_functions["utility"]
    compute_marginal_utility = user_utility_functions["marginal_utility"]
    compute_inverse_marginal_utility = user_utility_functions[
        "inverse_marginal_utility"
    ]

    compute_value = partial(
        calc_current_value,
        compute_utility=compute_utility,
    )

    compute_beginning_of_period_wealth = partial(
        user_budget_constraint,
        options=options,
    )

    transition_function = exogenous_transition_function

    if options["n_discrete_choices"] == 1:
        compute_upper_envelope = _return_policy_and_value
    else:
        compute_upper_envelope = fast_upper_envelope_wrapper

    return (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_value,
        compute_beginning_of_period_wealth,
        compute_upper_envelope,
        transition_function,
    )


def calc_current_value(
    consumption: np.ndarray,
    next_period_value: np.ndarray,
    choice: int,
    params: Dict[str, float],
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
    utility = compute_utility(consumption, choice, params)
    value = utility + params["beta"] * next_period_value

    return value


def _return_policy_and_value(
    endog_grid, policy, value, expected_value_zero_savings, *args
):
    endog_grid = jnp.append(0, endog_grid)
    policy = jnp.append(0, policy)
    value = jnp.append(expected_value_zero_savings, value)
    return endog_grid, policy, policy, value
