import numpy as np

from dcegm.pre_processing.shared import get_smallest_int_type


def create_indexer_for_space(space, max_var_values=None):
    """Create indexer for space."""

    # Indexer has always unsigned data type with integers starting at zero
    # Leave one additional value for the invalid number
    data_type = get_smallest_int_type(space.shape[0] + 1)
    max_value = np.iinfo(data_type).max

    if max_var_values is None:
        max_var_values = np.max(space, axis=0)

    map_vars_to_index = np.full(
        max_var_values + 1, fill_value=max_value, dtype=data_type
    )
    index_tuple = tuple(space[:, i] for i in range(space.shape[1]))

    map_vars_to_index[index_tuple] = np.arange(space.shape[0], dtype=data_type)

    return map_vars_to_index, max_value


def span_subspace(subdict_of_space, states_names):
    """Span subspace and read information from dictionary."""
    # Retrieve all state arrays from the dictionary
    states = [np.array(subdict_of_space[name]) for name in states_names]

    # Use np.meshgrid to get all combinations, then reshape and stack them
    grids = np.meshgrid(*states, indexing="ij")
    space = np.column_stack([grid.ravel() for grid in grids])

    return space
