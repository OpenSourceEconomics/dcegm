import numpy as np


def get_transition_matrix_by_state(state):
    """
    Return a transition matrix for each state.
    Args:
        child_state (np.ndarray): Array of shape (n_state_variables,) defining the
            agent's current child state.
    Returns:
        trans_mat (np.ndarray)

    """
    trans_mat = np.array([[1]])
    return trans_mat
