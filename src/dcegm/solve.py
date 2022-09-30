"""Interface for the DC-EGM algorithm."""
import copy
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.egm_step import do_egm_step
from dcegm.state_space import create_state_space
from dcegm.state_space import get_child_states
from dcegm.state_space import get_index_by_state
from dcegm.state_space import get_state_specific_choice_set
from dcegm.upper_envelope_step import do_upper_envelope_step
from scipy.special import roots_sh_legendre
from scipy.stats import norm


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
            [n_periods, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] of contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each time period and each discrete choice.
        - value (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_periods, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] of contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each time period and each discrete choice.
    """
    max_wealth = params.loc[("assets", "max_wealth"), "value"]
    n_periods = options["n_periods"]
    n_grid_wealth = options["grid_points_wealth"]
    n_quad_points = options["quadrature_points_stochastic"]
    sigma = params.loc[("shocks", "sigma"), "value"]

    savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    state_space, state_indexer = create_state_space(options)

    # Gauss-Legendre (shifted) quadrature over the interval [0,1].
    # Standard Gauss-Legendre quadrature (scipy.special.roots_legendre)
    # integrates over [-1, 1].
    quad_points, quad_weights = roots_sh_legendre(n_quad_points)
    quad_points_normal = norm.ppf(quad_points)

    exogenous_grid = {
        "savings": savings_grid,
        "quadrature_points": quad_points_normal * sigma,
        "quadrature_weights": quad_weights,
    }

    policy_arr, value_arr = _create_multi_dim_arrays(state_space, options)
    policy_arr, value_arr = solve_final_period(
        state_space,
        state_indexer,
        policy_arr,
        value_arr,
        savings_grid=savings_grid,
        params=params,
        options=options,
        compute_utility=utility_functions["utility"],
    )

    # Backwards induction from second to last period (T - 1)
    for period in range(n_periods - 2, -1, -1):

        subset_states = state_space[np.where(state_space[:, 0] == period)]

        for state in subset_states:

            current_state_index = get_index_by_state(state, state_indexer)
            child_nodes = get_child_states(state, state_space, state_indexer)

            current_state_index = get_index_by_state(state, indexer)
            child_nodes = get_child_states(state, state_space, indexer)
            
            for child_state in child_nodes:
                child_state_ind = get_index_by_state(child_state, state_indexer)
                next_period_policy = policy_arr[child_state_ind]
                next_period_value = value_arr[child_state_ind]
                child_node_choice_set = get_state_specific_choice_set(
                    child_state, state_space, state_indexer
                )

                (current_policy, current_value, expected_value,) = do_egm_step(
                    child_state,
                    child_node_choice_set,
                    params=params,
                    options=options,
                    exogenous_grid=exogenous_grid,
                    utility_functions=utility_functions,
                    next_period_policy=next_period_policy,
                    next_period_value=next_period_value,
                )

                if options["n_discrete_choices"] > 1:
                    (current_policy, current_value) = do_upper_envelope_step(
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


def solve_final_period(
    state_space: np.ndarray,
    indexer: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    savings_grid: np.ndarray,
    *,
    params: pd.DataFrame,
    options: Dict[str, int],
    compute_utility: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes solution to final period for policy and value function.

    Args:
        state_space (np.ndarray): Collection of all possible states.
        indexer (np.ndarray): Indexer object, that maps states to indexes.
        policy (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific policy function; of shape
            [n_periods, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] of contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each time period and each discrete choice.
        value (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_periods, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] of contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each time period and each discrete choice.
        savings_grid (np.ndarray): Array of shape (n_wealth_grid,) denoting the
            exogenous savings grid.
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        compute_utility (callable): Function for computation of agent's utility.

    Returns:
        (tuple): Tuple containing

        - policy (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific consumption policies with the solution for the final
            period included.
        - value (List[np.ndarray]): Nested list of np.ndarrays storing the
            choice-specific value functions with the solution for the final period
            included.
    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]

    # In last period, nothing is saved for the next period (since there is none).
    # Hence, everything is consumed, c_T(M, d) = M
    states_last_period = state_space[np.where(state_space[:, 0] == n_periods - 1)]

    end_grid = savings_grid.shape[0] + 1
    for state in states_last_period:
        state_index = indexer[state[0], state[1]]

        for choice in range(n_choices):
            policy[state_index, choice, 0, 1:end_grid] = copy.deepcopy(
                savings_grid
            )  # M
            policy[state_index, choice, 1, 1:end_grid] = copy.deepcopy(
                policy[state_index, choice, 0, 1:end_grid]
            )  # c(M, d)
            policy[state_index, choice, 0, 0] = 0
            policy[state_index, choice, 1, 0] = 0

            value[state_index, choice, 0, 2:end_grid] = compute_utility(
                policy[state_index, choice, 0, 2:end_grid], choice, params
            )
            value[state_index, choice][1, 2:end_grid] = compute_utility(
                policy[state_index, choice, 1, 2:end_grid], choice, params
            )
            value[state_index, choice, 0, 0] = 0
            value[state_index, choice, :, 2] = 0

    return policy, value


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
            [n_periods, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] of contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each time period and each discrete choice.
        - value (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_periods, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] of contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each time period and each discrete choice.
    """
    n_grid_wealth = options["grid_points_wealth"]
    n_choices = options["n_discrete_choices"]
    n_states = state_space.shape[0]

    policy_arr = np.empty((n_states, n_choices, 2, int(1.1 * n_grid_wealth)))
    value_arr = np.empty((n_states, n_choices, 2, int(1.1 * n_grid_wealth)))
    policy_arr[:] = np.nan
    value_arr[:] = np.nan

    return policy_arr, value_arr
