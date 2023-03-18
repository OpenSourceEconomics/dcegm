from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np


def final_period_wrapper(
    states: np.ndarray,
    savings_grid: np.ndarray,
    options: Dict[str, int],
    compute_utility: Callable,
    final_period_solution,  # noqa: U100
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes solution to final period for policy and value function.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        states (np.ndarray): Collection of all possible states.
        savings_grid (np.ndarray): Array of shape (n_wealth_grid,) denoting the
            exogenous savings grid.
        options (dict): Options dictionary.
        compute_utility (callable): Function for computation of agent's utility.

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
    n_choices = options["n_discrete_choices"]
    choice_range = [1] if n_choices < 2 else range(n_choices)
    n_states = states.shape[0]

    policy_final = np.empty(
        (n_states, n_choices, 2, int(1.1 * (len(savings_grid) + 1)))
    )
    value_final = np.empty((n_states, n_choices, 2, int(1.1 * (len(savings_grid) + 1))))
    policy_final[:] = np.nan
    value_final[:] = np.nan

    # In last period, nothing is saved for the next period (since there is none).
    # Hence, everything is consumed, c_T(M, d) = M
    for state_index in range(n_states):
        for choice_index, _ in enumerate(choice_range):
            for i, saving in enumerate(savings_grid):
                consumption, value = final_period_solution(
                    state=states[state_index],
                    begin_of_period_resources=saving,
                    options=options,
                    params_dict={},
                    compute_utility=compute_utility,
                )

                policy_final[state_index, choice_index, 0, i] = saving
                policy_final[state_index, choice_index, 1, i] = consumption

            value_final[state_index, choice_index, 0, : savings_grid.shape[0]] = (
                np.ones(savings_grid.shape[0]) * np.inf
            )
            value_final[
                state_index, choice_index, 1, : savings_grid.shape[0]
            ] = np.zeros(savings_grid.shape[0])

    return policy_final, value_final