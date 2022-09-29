import numpy as np


def create_state_space(options):
    """Create state space objects and indexer.

    Args:
        options (dict): Options dictionary.

    Returns:
        states (np.ndarray): Collection of all possible states.
        indexer (np.ndarray): Indexer object, that maps states to indexes.

    """
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]
    n_choices = 1 if n_choices < 2 else n_choices
    shape = (n_periods, n_choices)
    indexer = np.full(shape, -9999, dtype=np.int64)
    data = []
    i = 0

    for period in range(n_periods):
        for last_period_decision in range(n_choices):
            indexer[period, last_period_decision] = i
            row = [period, last_period_decision]
            i += 1
            data.append(row)

    states = np.array(data, dtype=np.int64)
    return states, indexer


def get_state_choice_set(
    state: np.ndarray,
    state_space: np.ndarray,
    indexer: np.ndarray,
):
    """Select choice set per state. Will be a user defined function later.
    This is very basic in Ishakov.

    Args:
        state (np.ndarray): Current individual state.
        state_space (np.ndarray): Collection of all possible states.
        indexer (np.ndarray): Indexer object, that maps states to indexes.

    Returns:
        choice_set (np.ndarray): This is the choice set in this state.

    """
    n_choices = indexer.shape[1]
    # If no discrete choices to make, set choice_range to 1 = "working".
    choice_set = [1] if n_choices < 2 else range(n_choices)
    return choice_set
