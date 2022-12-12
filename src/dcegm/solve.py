"""Interface for the DC-EGM algorithm."""
from __future__ import annotations

from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.egm_step import do_egm_step
from dcegm.pre_processing import create_multi_dim_arrays
from dcegm.pre_processing import get_partial_functions
from dcegm.state_space import get_child_states
from dcegm.upper_envelope_step import do_upper_envelope_step


def solve_dcegm(
    params: pd.DataFrame,
    options: dict[str, int],
    utility_functions: dict[str, Callable],
    budget_functions: dict[str, Callable],
    state_space_functions: dict[str, Callable],
    solve_final_period: Callable,
) -> tuple[np.ndarray, np.ndarray]:
    """Solves a discrete-continuous life-cycle model using the DC-EGM algorithm.

    EGM stands for Endogenous Grid Method.

    Args:
        params (pd.DataFrame): Model parameters indexed with multi-index of the
            form ("category", "name") and two columns ["value", "comment"].
        options (dict): Options dictionary.
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of:

            (i) utility
            (ii) inverse marginal utility
            (iii) next period marginal utility
        budget_functions (Dict[str, callable]): Dictionary of two user-supplied
            functions for computation of:

            (i) the budget constraint
            (ii) marginal budget constraint with respect to end of period assets
        state_space_functions (Dict[str, callable]): Dictionary of two user-supplied
            functions to:

            (i) create the state space
            (ii) get the state specific choice set
        solve_final_period (callable): User-supplied function for solving the agent's
            last period.

     Returns:
        (tuple): Tuple containing:

        - policy (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific policy function; of shape
            [n_states, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each state and each discrete choice.

        - value (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_states, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each state and each discrete choice.

    """
    max_wealth = params.loc[("assets", "max_wealth"), "value"]
    n_periods = options["n_periods"]
    n_grid_wealth = options["grid_points_wealth"]

    exogenous_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    create_state_space = state_space_functions["create_state_space"]
    get_state_specific_choice_set = state_space_functions[
        "get_state_specific_choice_set"
    ]
    state_space, state_indexer = create_state_space(options)
    policy_arr, value_arr = create_multi_dim_arrays(state_space, options)

    (
        compute_utility,
        compute_marginal_utility,
        compute_current_policy,
        compute_current_value,
        compute_expected_value,
        compute_next_choice_probs,
        compute_next_wealth_matrices,
        compute_next_marginal_wealth,
        store_current_policy_and_value,
    ) = get_partial_functions(
        params,
        options,
        exogenous_savings_grid,
        user_utility_func=utility_functions["utility"],
        user_marginal_utility_func=utility_functions["marginal_utility"],
        user_inverse_marginal_utility_func=utility_functions[
            "inverse_marginal_utility"
        ],
        user_budget_constraint=budget_functions["budget_constraint"],
        user_marginal_next_period_wealth=budget_functions["marginal_budget_constraint"],
    )

    _state_indices_final_period = np.where(state_space[:, 0] == n_periods - 1)
    states_final_period = state_space[_state_indices_final_period]
    policy_final, value_final = solve_final_period(
        states=states_final_period,
        savings_grid=exogenous_savings_grid,
        options=options,
        compute_utility=compute_utility,
    )

    policy_arr[_state_indices_final_period, ...] = policy_final
    value_arr[_state_indices_final_period, ...] = value_final

    for period in range(n_periods - 2, -1, -1):

        state_subspace = state_space[np.where(state_space[:, 0] == period)]

        for state in state_subspace:

            current_state_index = state_indexer[tuple(state)]
            child_nodes = get_child_states(
                state,
                state_space,
                state_indexer,
                get_state_specific_choice_set=get_state_specific_choice_set,
            )

            for child_state in child_nodes:

                child_state_index = state_indexer[tuple(child_state)]

                next_period_policy = policy_arr[child_state_index]
                next_period_value = value_arr[child_state_index]

                child_node_choice_set = get_state_specific_choice_set(
                    child_state, state_space, state_indexer
                )

                current_policy, current_value = do_egm_step(
                    child_state,
                    child_node_choice_set,
                    options=options,
                    compute_marginal_utility=compute_marginal_utility,
                    compute_current_policy=compute_current_policy,
                    compute_current_value=compute_current_value,
                    compute_expected_value=compute_expected_value,
                    compute_next_choice_probs=compute_next_choice_probs,
                    compute_next_wealth_matrices=compute_next_wealth_matrices,
                    compute_next_marginal_wealth=compute_next_marginal_wealth,
                    store_current_policy_and_value=store_current_policy_and_value,
                    next_policy=next_period_policy,
                    next_value=next_period_value,
                )

                if options["n_discrete_choices"] > 1:
                    compute_value = partial(
                        compute_current_value,
                        choice=child_state[1],  # working decision
                    )
                    current_policy, current_value = do_upper_envelope_step(
                        current_policy,
                        current_value,
                        options=options,
                        compute_value=compute_value,
                    )

                # Store
                policy_arr[
                    current_state_index,
                    child_state[1],
                    :,
                    : current_policy.shape[1],
                ] = current_policy
                value_arr[
                    current_state_index,
                    child_state[1],
                    :,
                    : current_value.shape[1],
                ] = current_value

    return policy_arr, value_arr
