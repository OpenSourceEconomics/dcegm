"""User-defined functions for creating state space objects."""
from typing import Dict
from typing import Tuple

import numpy as np


def create_state_space(options: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create state space object and indexer.

    We need to add the convention for the state space objects.

    Args:
        options (dict): Options dictionary.

    Returns:
        tuple:

        - state_vars (list): List of state variables.
        - state_space (np.ndarray): 2d array of shape (n_states, n_state_variables + 1)
            which serves as a collection of all possible states. By convention,
            the first column must contain the period and the last column the
            exogenous processes. Any other state variables are in between.
            E.g. if the two state variables are period and lagged choice and all choices
            are admissible in each period, the shape of the state space array is
            (n_periods * n_choices, 3).
        - map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
            The shape of this object is quite complicated. For each state variable it
            has the number of possible states as rows, i.e.
            (n_poss_states_state_var_1, n_poss_states_state_var_2, ....).

    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]  # lagged_choice is a state variable
    n_exog_states = options["n_exog_states"]

    shape = (n_periods, n_choices, n_exog_states)

    map_state_to_index = np.full(shape, -9999, dtype=np.int64)
    _state_space = []

    i = 0
    for period in range(n_periods):
        for lagged_choice in range(n_choices):
            for exog_state in range(n_exog_states):
                map_state_to_index[period, lagged_choice, exog_state] = i

                row = [period, lagged_choice, exog_state]
                _state_space.append(row)

                i += 1

    state_space = np.array(_state_space, dtype=np.int64)

    return state_space, map_state_to_index


def create_state_space_two_exog_processes(
    options: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Create state space object and indexer.

    We need to add the convention for the state space objects.

    Args:
        options (dict): Options dictionary.

    Returns:
        tuple:

        - state_vars (list): List of state variables.
        - state_space (np.ndarray): 2d array of shape (n_states, n_state_variables + 1)
            which serves as a collection of all possible states. By convention,
            the first column must contain the period and the last column the
            exogenous processes. Any other state variables are in between.
            E.g. if the two state variables are period and lagged choice and all choices
            are admissible in each period, the shape of the state space array is
            (n_periods * n_choices, 3).
        - map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
            The shape of this object is quite complicated. For each state variable it
            has the number of possible states as rows, i.e.
            (n_poss_states_state_var_1, n_poss_states_state_var_2, ....).

    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]  # lagged_choice is a state variable
    n_exog_one = options["n_exog_one"]
    n_exog_two = options["n_exog_two"]

    n_married = 2

    shape = (
        n_periods,
        n_periods,
        n_choices,
        n_exog_one * n_exog_two,
        # n_exog_one * n_exog_two,
    )

    map_state_to_index = np.full(shape, -9999, dtype=np.int64)
    _state_space = []

    i = 0
    for period in range(n_periods):
        for married in range(n_married):
            # for exog1 in range(n_exog_one):
            #     for exog2 in range(n_exog_two):
            for lagged_choice in range(n_choices):
                for lagged_exog in range(n_exog_one * n_exog_two):
                    # for exog in range(n_exog_one * n_exog_two):
                    map_state_to_index[period, married, lagged_choice, lagged_exog] = i

                    row = [period, married, lagged_choice, lagged_exog]
                    _state_space.append(row)

                    i += 1

    state_space = np.array(_state_space, dtype=np.int64)

    return state_space, map_state_to_index


def update_state(state, choice):
    """Get endogenous state by state and choice.

    Args:
        state (np.ndarray): 1d array of shape (n_state_vars,) containing the
            current state.
        choice (int): Choice to be made at the end of the period.

    Returns:
        np.ndarray: 1d array of shape (n_state_vars,) containing the state of the
            next period, where the endogenous part of the state is updated.

    """
    state_next = state.copy()

    state_next[0] += 1
    state_next[1] = choice

    return state_next


def get_state_specific_feasible_choice_set(
    state: np.ndarray,
    map_state_to_state_space_index: np.ndarray,
) -> np.ndarray:
    """Select state-specific feasible choice set.

    Will be a user defined function later.

    This is very basic in Ishkakov et al (2017).

    Args:
        state (np.ndarray): Array of shape (n_state_variables,) defining the agent's
            state. In Ishkakov, an agent's state is defined by her (i) age (i.e. the
            current period) and (ii) her lagged labor market choice.
            Hence n_state_variables = 2.
        map_state_to_state_space_index (np.ndarray): Indexer array that maps
            a period-specific state vector to the respective index positions in the
            state space.
            The shape of this object is quite complicated. For each state variable it
            has the number of potential states as rows, i.e.
            (n_potential_states_state_var_1, n_potential_states_state_var_2, ....).

    Returns:
        choice_set (np.ndarray): 1d array of length (n_feasible_choices,) with the
            agent's (restricted) feasible choice set in the given state.

    """
    # lagged_choice is a state variable
    n_choices = map_state_to_state_space_index.shape[1]

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if state[1] == 1:
        feasible_choice_set = np.array([1])
    else:
        feasible_choice_set = np.arange(n_choices)

    return feasible_choice_set


def get_feasible_choice_set_two_exog_processes(
    state: np.ndarray,
    map_state_to_state_space_index: np.ndarray,
) -> np.ndarray:
    """Select state-specific feasible choice set.

    Will be a user defined function later.

    This is very basic in Ishkakov et al (2017).

    Args:
        state (np.ndarray): Array of shape (n_state_variables,) defining the agent's
            state. In Ishkakov, an agent's state is defined by her (i) age (i.e. the
            current period) and (ii) her lagged labor market choice.
            Hence n_state_variables = 2.
        map_state_to_state_space_index (np.ndarray): Indexer array that maps
            a period-specific state vector to the respective index positions in the
            state space.
            The shape of this object is quite complicated. For each state variable it
            has the number of potential states as rows, i.e.
            (n_potential_states_state_var_1, n_potential_states_state_var_2, ....).

    Returns:
        choice_set (np.ndarray): 1d array of length (n_feasible_choices,) with the
            agent's (restricted) feasible choice set in the given state.

    """
    # lagged_choice is a state variable
    n_choices = map_state_to_state_space_index.shape[1]

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if state[1] == 1:
        feasible_choice_set = np.array([1])
    else:
        feasible_choice_set = np.arange(n_choices)

    return feasible_choice_set
