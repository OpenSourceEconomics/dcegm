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
    shape = (
        n_periods,
        n_choices,
    )
    indexer = np.full(shape, -9999, dtype=np.int64)
    data = []
    indexer[0, 0] = 0
    i = 1

    for period in range(n_periods):
        # for last_period_decision in range(n_choices):
        #     indexer[period, last_period_decision] = i

        row = [period]
        i += 1
        data.append(row)

    states = np.array(data, dtype=np.int64)
    return states, indexer
