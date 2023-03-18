from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
from jax import vmap


def final_period_wrapper(
    final_period_states: np.ndarray,
    savings_grid: np.ndarray,
    options: Dict[str, int],
    compute_utility: Callable,
    final_period_solution,  # noqa: U100
    choices_child,
    compute_next_period_wealth,
    compute_marginal_utility,
    compute_value,
    taste_shock_scale,
    exogenous_savings_grid,
    income_shock_draws,
    income_shock_weights,
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
    n_states = final_period_states.shape[0]

    resources_last_period = vmap(
        vmap(
            vmap(compute_next_period_wealth, in_axes=(None, None, 0)),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, None, None),
    )(final_period_states, exogenous_savings_grid, income_shock_draws)

    policy_final = np.empty((n_states, n_choices, 2, savings_grid.shape[0]))
    value_final = np.empty((n_states, n_choices, 2, savings_grid.shape[0]))

    final_period_solution_partial = partial(
        final_period_solution,
        options=options,
        params_dict={},
        compute_utility=compute_utility,
    )
    # In last period, nothing is saved for the next period (since there is none).
    # Hence, everything is consumed, c_T(M, d) = M
    for state_index in range(n_states):
        for i, saving in enumerate(savings_grid):
            for choice in range(n_choices):
                consumption, value = final_period_solution_partial(
                    state=final_period_states[state_index],
                    begin_of_period_resources=saving,
                    choice=choice,
                )

                policy_final[state_index, choice, 0, i] = saving
                policy_final[state_index, choice, 1, i] = consumption

                value_final[state_index, choice, 0, i] = saving
                value_final[state_index, choice, 1, i] = value

        # (
        #     marginal_utilities[final_state_cond, :],
        #     max_expected_values[final_state_cond, :],
        # ) = marginal_util_and_exp_max_value_states_period_jitted(
        #     possible_child_states=states_final_period,
        #     choices_child_states=choices_child_states,
        #     policies_child_states=policy_final,
        #     values_child_states=value_final,
        # )

    return policy_final, value_final
