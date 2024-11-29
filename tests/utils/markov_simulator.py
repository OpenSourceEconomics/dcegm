import numpy as np


def markov_simulator(n_periods_to_sim, initial_dist, trans_probs):
    """Simulate a Markov process."""
    n_states = initial_dist.shape[0]
    final_dist = np.zeros((n_periods_to_sim, n_states))
    final_dist[0, :] = initial_dist

    for t in range(n_periods_to_sim - 1):
        current_dist = final_dist[t, :]
        for state in range(n_states - 1):
            final_dist[t + 1, state] = current_dist @ trans_probs[:, state]

        final_dist[t + 1, -1] = 1 - final_dist[t + 1, :-1].sum()

    return final_dist
