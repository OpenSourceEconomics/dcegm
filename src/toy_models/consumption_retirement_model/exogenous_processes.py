import numpy as np


def get_transition_matrix_by_state(state, params_dict):  # noqa: U100
    """Return a transition matrix for each state.

    Args:
        state (np.ndarray): Array of shape (n_state_variables,) defining the
            agent's current child state.
        params_dict (dict): Dictionary containing model parameters.

    Returns:
        trans_vec (np.ndarray): A vector containing for each possible exogenous
            process state the corresponding probability.
            Shape is (n_exog_processes).

    """
    trans_vec = np.array([1])
    return trans_vec
