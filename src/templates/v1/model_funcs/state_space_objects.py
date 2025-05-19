"""User-defined functions for creating state space objects."""

from typing import Dict

import numpy as np


def create_state_space_function_dict():
    """Create dictionary with state space functions.

    Returns:
        state_space_functions (dict): Dictionary with state space functions.

    """
    return {
        "state_specific_choice_set": get_state_specific_feasible_choice_set,
    }


def get_state_specific_feasible_choice_set(
    lagged_choice: int,
    model_specs: Dict,
) -> np.ndarray:
    """Select state-specific feasible choice set such that retirement is absorbing.

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
    n_choices = model_specs["n_choices"]

    # n_choices = len(model_specs["state_space"]["choices"])

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if lagged_choice == 1:
        feasible_choice_set = np.array([1])
    else:
        feasible_choice_set = np.arange(n_choices)

    return feasible_choice_set
