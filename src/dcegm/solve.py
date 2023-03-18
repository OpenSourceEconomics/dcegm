"""Interface for the DC-EGM algorithm."""
from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.egm import compute_optimal_policy_and_value
from dcegm.fast_upper_envelope import fast_upper_envelope_wrapper
from dcegm.final_period import final_period_wrapper
from dcegm.integration import quadrature_legendre
from dcegm.marg_utilities_and_exp_value import (
    marginal_util_and_exp_max_value_states_period,
)
from dcegm.pre_processing import create_multi_dim_arrays
from dcegm.pre_processing import get_partial_functions
from dcegm.pre_processing import get_possible_choices_array
from dcegm.pre_processing import params_todict
from dcegm.state_space import get_child_indexes
from jax import jit


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable],
    final_period_solution: Callable,
    user_transition_function: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve a discrete-continuous life-cycle model using the DC-EGM algorithm.

    Args:
        params_dict (dict): Dictionary containing model parameters.
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
        final_period_wrapper (callable): User-supplied function for solving the agent's
            last period.
        user_transition_function (callable): User-supplied function returning for each
            state a transition matrix vector.

    Returns:
        tuple:

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
    compute_upper_envelope = fast_upper_envelope_wrapper

    params_dict = params_todict(params)

    taste_shock_scale = params_dict["lambda"]
    interest_rate = params_dict["interest_rate"]
    discount_factor = params_dict["beta"]
    max_wealth = params_dict["max_wealth"]

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
        options["quadrature_points_stochastic"], params_dict["sigma"]
    )

    (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_value,
        compute_next_period_wealth,
        transition_vector_by_state,
    ) = get_partial_functions(
        params_dict,
        options,
        user_utility_functions=utility_functions,
        user_budget_constraint=budget_constraint,
        exogenous_transition_function=user_transition_function,
    )

    final_period_partial = partial(
        final_period_wrapper,
        options=options,
        compute_utility=compute_utility,
        final_period_solution=final_period_solution,
    )

    choice_set_array = get_possible_choices_array(
        state_space,
        state_indexer,
        state_space_functions["get_state_specific_choice_set"],
        options,
    )

    policy_array, value_array = create_multi_dim_arrays(state_space, options)

    policy_array, value_array = backwards_induction(
        n_periods,
        taste_shock_scale,
        discount_factor,
        interest_rate,
        state_indexer,
        state_space,
        income_shock_draws,
        income_shock_weights,
        exogenous_savings_grid,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_value,
        compute_next_period_wealth,
        get_state_specific_choice_set,
        choice_set_array,
        transition_vector_by_state,
        policy_array,
        value_array,
        compute_upper_envelope=compute_upper_envelope,
        final_period_partial=final_period_partial,
    )

    return policy_array, value_array


def backwards_induction(
    n_periods: int,
    taste_shock_scale: float,
    discount_factor: float,
    interest_rate: float,
    state_indexer: np.ndarray,
    state_space: np.ndarray,
    income_shock_draws: np.ndarray,
    income_shock_weights: np.ndarray,
    exogenous_savings_grid: np.ndarray,
    compute_marginal_utility: Callable,
    compute_inverse_marginal_utility: Callable,
    compute_value: Callable,
    compute_next_period_wealth: Callable,
    get_state_specific_choice_set: Callable,
    choice_set_array,
    transition_vector_by_state: Callable,
    policy_array: np.ndarray,
    value_array: np.ndarray,
    compute_upper_envelope: Callable,
    final_period_partial,
):
    """Do backwards induction and solve for optimal policy and value function.

    Args:
        n_periods (int): Number of periods.
        taste_shock_scale (float): The taste shock scale.
        discount_factor (float): The discount factor.
        interest_rate (float): The interest rate of capital.
        state_indexer (np.ndarray): Indexer object that maps states to indexes.
            The shape of this object quite complicated. For each state variable it
             has the number of possible states as "row", i.e.
            (n_poss_states_statesvar_1, n_poss_states_statesvar_2, ....)
        state_space (np.ndarray): Collection of all possible states of shape
            (n_states, n_state_variables).
        income_shock_draws (np.ndarray): 1d array of shape (n_quad_points,) containing
            the Hermite quadrature points.
        income_shock_weights (np.ndarrray): Weights for each stoachstic shock draw.
            Shape is (n_stochastic_quad_points)
        exogenous_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the exogenous savings grid.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        compute_inverse_marginal_utility (Callable): Function for calculating the
            inverse marginal utiFality, which takes the marginal utility as only input.
        compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
        compute_next_period_wealth (callable): User-defined function to compute the
            agent's wealth of the next period (t + 1). The inputs
            ```saving```, ```shock```, ```params``` and ```options```
            are already partialled in.
        get_state_specific_choice_set (Callable): User-supplied function returning for
            each state all possible choices.
        transition_vector_by_state (Callable): Partialled transition function return
            transition vector for each state.
        policy_array (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific policy function; of shape
            [n_states, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the policy function
            c(M, d), for each state and each discrete choice.
        value_array (np.ndarray): Multi-dimensional np.ndarray storing the
            choice-specific value functions; of shape
            [n_states, n_discrete_choices, 2, 1.1 * n_grid_wealth].
            Position [.., 0, :] contains the endogenous grid over wealth M,
            and [.., 1, :] stores the corresponding value of the value function
            v(M, d), for each state and each discrete choice.

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
    marginal_utilities = np.full(
        shape=(
            state_space.shape[0],
            exogenous_savings_grid.shape[0],
        ),
        fill_value=np.nan,
        dtype=float,
    )
    max_expected_values = np.full(
        shape=(state_space.shape[0], exogenous_savings_grid.shape[0]),
        fill_value=np.nan,
        dtype=float,
    )
    marginal_util_and_exp_max_value_states_period_jitted = jit(
        partial(
            marginal_util_and_exp_max_value_states_period,
            compute_next_period_wealth=compute_next_period_wealth,
            compute_marginal_utility=compute_marginal_utility,
            compute_value=compute_value,
            taste_shock_scale=taste_shock_scale,
            exogenous_savings_grid=exogenous_savings_grid,
            income_shock_draws=income_shock_draws,
            income_shock_weights=income_shock_weights,
        )
    )

    final_state_cond = np.where(state_space[:, 0] == n_periods - 1)[0]
    states_final_period = state_space[final_state_cond]

    (
        resources_final,
        policy_array[final_state_cond, :, 1, : exogenous_savings_grid.shape[0]],
        value_array[final_state_cond, :, 1, : exogenous_savings_grid.shape[0]],
        marginal_utilities[final_state_cond, :],
        max_expected_values[final_state_cond, :],
    ) = final_period_partial(
        final_period_states=states_final_period,
        choices_final=choice_set_array[final_state_cond],
        compute_next_period_wealth=compute_next_period_wealth,
        compute_marginal_utility=compute_marginal_utility,
        taste_shock_scale=taste_shock_scale,
        exogenous_savings_grid=exogenous_savings_grid,
        income_shock_draws=income_shock_draws,
        income_shock_weights=income_shock_weights,
    )
    policy_array[
        final_state_cond, :, 0, : exogenous_savings_grid.shape[0]
    ] = resources_final
    value_array[
        final_state_cond, :, 0, : exogenous_savings_grid.shape[0]
    ] = resources_final

    for period in range(n_periods - 2, -1, -1):
        periods_state_cond = np.where(state_space[:, 0] == period)[0]
        state_subspace = state_space[periods_state_cond]

        for state in state_subspace:
            current_state_index = state_indexer[tuple(state)]
            # The choice set and the indexes are different/of different shape
            # for each state. For jax we should go over all states.
            # With numba we could easily guvectorize here over states and calculate
            # everything on the state level. This could be really fast! However, how do
            # treat the partial functions?
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
            max_expected_values_child_states = np.take(
                max_expected_values, child_states_indexes, axis=0
            )

            trans_vec_state = transition_vector_by_state(state)

            for choice_index, choice in enumerate(choice_set):
                current_policy, current_value = compute_optimal_policy_and_value(
                    marginal_utilities_child_states[choice_index, :],
                    max_expected_values_child_states[choice_index, :],
                    discount_factor,
                    interest_rate,
                    choice,
                    trans_vec_state,
                    exogenous_savings_grid,
                    compute_inverse_marginal_utility=compute_inverse_marginal_utility,
                    compute_value=compute_value,
                )

                if policy_array.shape[1] > 1:
                    # For the upper envelope we cannot parallelize over the wealth grid
                    # as here we need to inspect the value function on the whole wealth
                    # grid.
                    current_policy, current_value = compute_upper_envelope(
                        policy=current_policy,
                        value=current_value,
                        exog_grid=exogenous_savings_grid,
                        choice=choice,
                        compute_value=compute_value,
                    )

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

            possible_child_states = state_space[periods_state_cond]
            policies_child_states = policy_array[periods_state_cond]
            values_child_states = value_array[periods_state_cond]
            choices_child_states = choice_set_array[periods_state_cond]

            (
                marginal_utilities[periods_state_cond, :],
                max_expected_values[periods_state_cond, :],
            ) = marginal_util_and_exp_max_value_states_period_jitted(
                possible_child_states=possible_child_states,
                choices_child_states=choices_child_states,
                policies_child_states=policies_child_states,
                values_child_states=values_child_states,
            )

    return policy_array, value_array
