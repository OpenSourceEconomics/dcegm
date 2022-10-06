"""Interface for the DC-EGM algorithm."""
from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.egm_step import do_egm_step
from dcegm.state_space import create_state_space
from dcegm.state_space import get_child_states
from dcegm.state_space import get_state_specific_choice_set
from dcegm.upper_envelope_step import do_upper_envelope_step
from scipy.special import roots_sh_legendre
from scipy.stats import norm
from toy_models.consumption_retirement_model import calc_current_period_policy
from toy_models.consumption_retirement_model import calc_expected_value
from toy_models.consumption_retirement_model import calc_next_period_choice_probs
from toy_models.consumption_retirement_model import calc_next_period_marginal_wealth
from toy_models.consumption_retirement_model import calc_next_period_wealth_matrices
from toy_models.consumption_retirement_model import calc_stochastic_income
from toy_models.consumption_retirement_model import calc_value_constrained
from toy_models.consumption_retirement_model import solve_final_period


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_functions: Dict[str, callable],
) -> Tuple[np.ndarray, np.ndarray]:
    """Solves a discrete-continuous life-cycle model using the DC-EGM algorithm.

    EGM stands for Endogenous Grid Method.

    Args:
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of:

            (i) utility
            (ii) inverse marginal utility
            (iii) next period marginal utility

     Returns:
        (tuple): Tuple containing

        - policy (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific policy function; of shape
            [n_states, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each state and each discrete choice.
        - value (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_states, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each state and each discrete choice.
    """
    max_wealth = params.loc[("assets", "max_wealth"), "value"]
    n_periods = options["n_periods"]
    n_grid_wealth = options["grid_points_wealth"]

    state_space, state_indexer = create_state_space(options)

    exogenous_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    policy_arr, value_arr = _create_multi_dim_arrays(state_space, options)

    condition_final_period = np.where(state_space[:, 0] == n_periods - 1)
    states_final_period = state_space[condition_final_period]

    (
        compute_utility,
        compute_marginal_utility,
        compute_current_policy,
        compute_value_constrained,
        compute_expected_value,
        compute_next_choice_probs,
        compute_next_wealth_matrices,
        compute_next_marginal_wealth,
        store_current_policy_and_value,
    ) = partial_functions(
        params,
        options,
        exogenous_savings_grid,
        utility_functions["utility"],
        utility_functions["marginal_utility"],
        utility_functions["inverse_marginal_utility"],
    )

    policy_final, value_final = solve_final_period(
        states=states_final_period,
        savings_grid=exogenous_savings_grid,
        options=options,
        compute_utility=compute_utility,
    )

    policy_arr[condition_final_period, ...] = policy_final
    value_arr[condition_final_period, ...] = value_final

    for period in range(n_periods - 2, -1, -1):

        state_subspace = state_space[np.where(state_space[:, 0] == period)]

        for state in state_subspace:

            current_state_index = state_indexer[tuple(state)]
            child_nodes = get_child_states(state, state_space, state_indexer)

            for child_state in child_nodes:

                child_state_index = state_indexer[tuple(child_state)]

                next_period_policy = policy_arr[child_state_index]
                next_period_value = value_arr[child_state_index]

                child_node_choice_set = get_state_specific_choice_set(
                    child_state, state_space, state_indexer
                )

                current_policy, current_value, expected_value = do_egm_step(
                    child_state,
                    child_node_choice_set,
                    options=options,
                    compute_utility=compute_utility,
                    compute_marginal_utility=compute_marginal_utility,
                    compute_current_policy=compute_current_policy,
                    compute_value_constrained=compute_value_constrained,
                    compute_expected_value=compute_expected_value,
                    compute_next_choice_probs=compute_next_choice_probs,
                    compute_next_wealth_matrices=compute_next_wealth_matrices,
                    compute_next_marginal_wealth=compute_next_marginal_wealth,
                    store_current_policy_and_value=store_current_policy_and_value,
                    next_policy=next_period_policy,
                    next_value=next_period_value,
                )

                if options["n_discrete_choices"] > 1:
                    current_policy, current_value = do_upper_envelope_step(
                        current_policy,
                        current_value,
                        expected_value=expected_value,
                        params=params,
                        options=options,
                        compute_utility=utility_functions["utility"],
                    )

                # Store
                policy_arr[
                    current_state_index,
                    child_state[1],
                    :,
                    : current_policy.shape[1],
                ] = current_policy
                value_arr[
                    current_state_index,
                    child_state[1],
                    :,
                    : current_value.shape[1],
                ] = current_value

    return policy_arr, value_arr


def partial_functions(
    params,
    options,
    exogenous_savings_grid,
    user_utility_func,
    user_marginal_utility_func,
    user_inverse_marginal_utility_func,
):
    sigma = params.loc[("shocks", "sigma"), "value"]
    n_quad_points = options["quadrature_points_stochastic"]
    # Gauss-Legendre (shifted) quadrature over the interval [0,1].
    quad_points, quad_weights = roots_sh_legendre(n_quad_points)
    quad_points_normal = norm.ppf(quad_points)
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
    compute_income = partial(
        calc_stochastic_income,
        wage_shock=quad_points_normal * sigma,
        params=params,
        options=options,
    )
    compute_current_policy = partial(
        calc_current_period_policy,
        quad_weights=quad_weights,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
    )
    compute_value_constrained = partial(
        calc_value_constrained,
        beta=params.loc[("beta", "beta"), "value"],
        compute_utility=compute_utility,
    )
    compute_expected_value = partial(
        calc_expected_value,
        params=params,
        quad_weights=quad_weights,
    )
    compute_next_choice_probs = partial(
        calc_next_period_choice_probs, params=params, options=options
    )
    compute_next_wealth_matrices = partial(
        calc_next_period_wealth_matrices,
        savings=exogenous_savings_grid,
        params=params,
        options=options,
        compute_income=compute_income,
    )
    compute_next_marginal_wealth = partial(
        calc_next_period_marginal_wealth,
        params=params,
        options=options,
    )
    store_current_policy_and_value = partial(
        _store_current_period_policy_and_value,
        savings=exogenous_savings_grid,
        params=params,
        options=options,
        compute_utility=compute_utility,
    )
    return (
        compute_utility,
        compute_marginal_utility,
        compute_current_policy,
        compute_value_constrained,
        compute_expected_value,
        compute_next_choice_probs,
        compute_next_wealth_matrices,
        compute_next_marginal_wealth,
        store_current_policy_and_value,
    )


def _store_current_period_policy_and_value(
    current_period_policy: np.ndarray,
    expected_value: np.ndarray,
    child_state: np.ndarray,
    savings: np.ndarray,
    params: pd.DataFrame,
    options: Dict[str, int],
    compute_utility: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Store the current period policy and value funtions.

    Args:
        current_period_policy (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the agent's current period policy rule.
        expected_value (np.ndarray): (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the agent's expected value of the next period.
        child_state (np.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        savings (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            exogenous savings grid .
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        compute_utility (callable): User-defined function to compute the agent's
            utility. The input ```params``` is already partialled in.

    Returns:
        (tuple): Tuple containing:

        - current_policy (np.ndarray): 2d array of the agent's period- and
            choice-specific consumption policy. Shape (2, 1.1 * (n_grid_wealth + 1)).
            Position [0, :] contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the policy function c(M, d).
        - current_value (np.ndarray): 2d array of the agent's period- and
            choice-specific value function. Shape (2, 1.1 * (n_grid_wealth + 1)).
            Position [0, :] contains the endogenous grid over wealth M,
            and [1, :] stores the corresponding value of the value function v(M, d).

    """
    beta = params.loc[("beta", "beta"), "value"]
    n_grid_wealth = options["grid_points_wealth"]

    endogenous_wealth_grid = savings + current_period_policy

    current_period_utility = compute_utility(current_period_policy, child_state[1])

    current_policy = np.zeros((2, n_grid_wealth + 1))
    current_policy[0, 1:] = endogenous_wealth_grid
    current_policy[1, 1:] = current_period_policy

    current_value = np.zeros((2, n_grid_wealth + 1))
    current_value[0, 1:] = endogenous_wealth_grid
    current_value[1, 0] = expected_value[0]
    current_value[1, 1:] = current_period_utility + beta * expected_value

    return current_policy, current_value


def _create_multi_dim_arrays(
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
    n_grid_wealth = options["grid_points_wealth"]
    n_choices = options["n_discrete_choices"]
    n_states = state_space.shape[0]

    policy_arr = np.empty((n_states, n_choices, 2, int(1.1 * n_grid_wealth + 1)))
    value_arr = np.empty((n_states, n_choices, 2, int(1.1 * n_grid_wealth + 1)))
    policy_arr[:] = np.nan
    value_arr[:] = np.nan

    return policy_arr, value_arr
