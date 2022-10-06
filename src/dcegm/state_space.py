from typing import Dict
from typing import Tuple

import numpy as np


def create_state_space(options: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create state space object and indexer.

    Args:
        options (dict): Options dictionary.

    Returns:
        state_space (np.ndarray): Collection of all possible states of shape
            (n_periods * n_choices, n_choices).
        indexer (np.ndarray): Indexer object that maps states to indexes.
            Shape (n_periods, n_choices).

    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]

    shape = (n_periods, n_choices)
    indexer = np.full(shape, -9999, dtype=np.int64)

    _state_space = []

    i = 0
    for period in range(n_periods):
        for last_period_decision in range(n_choices):
            indexer[period, last_period_decision] = i

            row = [period, last_period_decision]
            _state_space.append(row)

            i += 1

    state_space = np.array(_state_space, dtype=np.int64)

    return state_space, indexer


def get_state_specific_choice_set(
    state: np.ndarray,
    state_space: np.ndarray,
    indexer: np.ndarray,
) -> np.ndarray:
    """Select state-specific choice set. Will be a user defined function later.

    This is very basic in Ishkakov.

    Args:
        state (np.ndarray): Array of shape (n_state_variables,) defining the agent's
            state. In Ishkakov, an agent's state is defined by her (i) age (i.e. the
            current period) and (ii) her lagged labor market choice.
            Hence n_state_variables = 2.
        state_space (np.ndarray): Collection of all possible states of shape
            (n_periods * n_choices, n_choices).
        indexer (np.ndarray): Indexer object that maps states to indexes.
            Shape (n_periods, n_choices).

    Returns:
        choice_set (np.ndarray): The agent's (restricted) choice set in the given
            state of shape (n_admissible_choices,).

    """
    n_state_variables = indexer.shape[1]

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if state[1] == 1:
        choice_set = np.array([1])
    else:
        choice_set = np.arange(n_state_variables)

    return choice_set


def get_child_states(
    state: np.ndarray,
    state_space: np.ndarray,
    indexer: np.ndarray,
) -> np.ndarray:
    """Select state-specific child nodes. Will be a user defined function later.

    Args:
        # state (np.ndarray): Array of shape (n_state_variables,) defining the agent's
            state. In Ishkakov, an agent's state is defined by her (i) age (i.e. the
            current period) and (ii) her lagged labor market choice.
            Hence n_state_variables = 2.
        states (np.ndarray): Collection of all possible states of shape
            (n_periods * n_choices, n_choices).
        indexer (np.ndarray): Indexer object that maps states to indexes.
            Shape (n_periods, n_choices).

    Returns:
        child_nodes (np.ndarray): Array of child nodes the agent can reach from the
            given state. Shape (n_state_specific_choices, n_state_specific_choices).

    """
    # Child nodes are so far n_choices by state_space variables.
    state_specific_choice_set = get_state_specific_choice_set(
        state, state_space, indexer
    )
    child_nodes = np.empty(
        (state_specific_choice_set.shape[0], state_space.shape[1]), dtype=int
    )  # (n_admissible_choices, n_state_variables)

    for i, choice in enumerate(state_specific_choice_set):
        child_nodes[i, :] = state_space[indexer[state[0] + 1, choice]]

    return child_nodes


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
