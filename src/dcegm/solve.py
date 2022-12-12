"""Interface for the DC-EGM algorithm."""
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.egm_step import compute_optimal_policy_and_value
from dcegm.egm_step import get_child_state_policy_and_value
from dcegm.integration import quadrature_legendre
from dcegm.pre_processing import create_multi_dim_arrays
from dcegm.pre_processing import get_partial_functions
from dcegm.state_space import get_child_indexes
from dcegm.upper_envelope_step import do_upper_envelope_step


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable],
    solve_final_period: Callable,
    transition_vector_by_state: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
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
        budget_constraint (callable): Callable budget constraint.
        state_space_functions (Dict[str, callable]): Dictionary of two user-supplied
            functions to:

            (i) create the state space
            (ii) get the state specific choice set
        solve_final_period (callable): User-supplied function for solving the agent's
            last period.
        transition_vector_by_state (callable): User-supplied function returning for each
            state a transition matrix vector.

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
    # ToDo: Make interface with several draw possibilities.
    # ToDo: Some day make user supplied draw function.
    income_shock_draws, income_shock_weights = quadrature_legendre(
        options["quadrature_points_stochastic"],
        params.loc[("shocks", "sigma"), "value"],
    )

    (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_value,
        compute_next_wealth_matrices,
    ) = get_partial_functions(
        params,
        options,
        user_utility_functions=utility_functions,
        user_budget_constraint=budget_constraint,
    )

    _state_indices_final_period = np.where(state_space[:, 0] == n_periods - 1)
    states_final_period = state_space[_state_indices_final_period]
    policy_final, value_final = solve_final_period(
        states=states_final_period,
        savings_grid=exogenous_savings_grid,
        options=options,
        compute_utility=compute_utility,
    )

    taste_shock_scale = params.loc[("shocks", "lambda"), "value"]
    interest_rate = params.loc[("assets", "interest_rate"), "value"]

    policy_array, value_array = create_multi_dim_arrays(state_space, options)
    policy_array[_state_indices_final_period, ...] = policy_final
    value_array[_state_indices_final_period, ...] = value_final

    policy_array, value_array = backwards_induction(
        n_periods,
        taste_shock_scale,
        interest_rate,
        state_indexer,
        state_space,
        income_shock_draws,
        income_shock_weights,
        exogenous_savings_grid,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_value,
        compute_next_wealth_matrices,
        get_state_specific_choice_set,
        transition_vector_by_state,
        policy_array,
        value_array,
    )

    return policy_array, value_array


def backwards_induction(
    n_periods,
    taste_shock_scale,
    interest_rate,
    state_indexer,
    state_space,
    income_shock_draws,
    income_shock_weights,
    exogenous_savings_grid,
    compute_marginal_utility,
    compute_inverse_marginal_utility,
    compute_value,
    compute_next_wealth_matrices,
    get_state_specific_choice_set,
    transition_vector_by_state,
    policy_array,
    value_array,
):
    marginal_utilities = np.full(
        shape=(
            state_space.shape[0],
            exogenous_savings_grid.shape[0] * income_shock_weights.shape[0],
        ),
        fill_value=np.nan,
        dtype=float,
    )
    max_expected_values = np.full(
        shape=(
            state_space.shape[0],
            exogenous_savings_grid.shape[0] * income_shock_weights.shape[0],
        ),
        fill_value=np.nan,
        dtype=float,
    )
    for period in range(n_periods - 2, -1, -1):

        possible_child_states = state_space[np.where(state_space[:, 0] == period + 1)]
        for child_state in possible_child_states:
            child_state_index = state_indexer[tuple(child_state)]
            # We could parralelize here also over the savings grid!

            (
                marginal_utilities[child_state_index, :],
                max_expected_values[child_state_index, :],
            ) = get_child_state_policy_and_value(
                exogenous_savings_grid,
                income_shock_draws,
                child_state,
                state_indexer,
                state_space,
                taste_shock_scale,
                policy_array,
                value_array,
                compute_next_wealth_matrices,
                compute_marginal_utility,
                compute_value,
                get_state_specific_choice_set,
            )

        index_periods = np.where(state_space[:, 0] == period)[0]
        state_subspace = state_space[index_periods]
        for state in state_subspace:

            current_state_index = state_indexer[tuple(state)]
            # The choice set and the indexes are different/of different shape
            # for each state. For jax we should go over all states.
            # With numba we could easily guvectorize here over states and calculate
            # everything on the state level. This could be really fast! However, how do
            # treat the partial functiosn?
            choice_set = get_state_specific_choice_set(
                state, state_space, state_indexer
            )
            child_states_indexes = get_child_indexes(
                state, state_space, state_indexer, get_state_specific_choice_set
            )
            # With this next step is clear! Create transition vector, child indexes for
            # each state. For index use exception handling from take! Then create arrays
            # for all states. And then parralize over child, choice combinations for
            # upper envelope!
            marginal_utilities_child_states = np.take(
                marginal_utilities, child_states_indexes, axis=0
            )
            max_expected_values_child_states = max_expected_values[child_states_indexes]
            trans_vec_state = transition_vector_by_state(state)
            for choice_ind, choice in enumerate(choice_set):
                current_policy, current_value = compute_optimal_policy_and_value(
                    marginal_utilities_child_states[choice_ind, :],
                    max_expected_values_child_states[choice_ind, :],
                    interest_rate,
                    choice,
                    income_shock_weights,
                    trans_vec_state,
                    exogenous_savings_grid,
                    compute_inverse_marginal_utility=compute_inverse_marginal_utility,
                    compute_value=compute_value,
                )

                if policy_array.shape[1] > 1:
                    # For the upper envelope we cannot parralize over the wealth grid
                    # as here we need to inspect the value function on the whole wealth
                    # grid.
                    current_policy, current_value = do_upper_envelope_step(
                        current_policy,
                        current_value,
                        choice,
                        n_grid_wealth=exogenous_savings_grid.shape[0],
                        compute_value=compute_value,
                    )
                    if period == (n_periods - 5):
                        breakpoint()
                # Store
                policy_array[
                    current_state_index,
                    choice,
                    :,
                    : current_policy.shape[1],
                ] = current_policy
                value_array[
                    current_state_index,
                    choice,
                    :,
                    : current_value.shape[1],
                ] = current_value

    return policy_array, value_array
