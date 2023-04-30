"""Interface for the DC-EGM algorithm."""
from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.egm import compute_optimal_policy_and_value
from dcegm.final_period import final_period_wrapper
from dcegm.integration import quadrature_legendre
from dcegm.marg_utilities_and_exp_value import (
    marginal_util_and_exp_max_value_states_period,
)
from dcegm.pre_processing import convert_params_to_dict
from dcegm.pre_processing import create_multi_dim_arrays
from dcegm.pre_processing import get_partial_functions
from dcegm.state_space import create_state_space_admissible_choices
from dcegm.state_space import get_child_states_index
from dcegm.state_space import get_possible_choices_array
from jax import jit
from jax import vmap


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict[str, int],
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable],
    final_period_solution: Callable,
    transition_function: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a discrete-continuous life-cycle model using the DC-EGM algorithm.

    Args:
        params (pd.DataFrame): Params DataFrame.
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
        final_period_solution (callable): User-supplied function for solving the agent's
            last period.
        transition_function (callable): User-supplied function returning for each
            state a transition matrix vector.

    Returns:
        tuple:

        - endog_grid_container (np.ndarray): "Filled" 3d array containing the
            endogenous grid for each state and each discrete choice.
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
        - policy_container (np.ndarray): "Filled" 3d array containing the
            choice-specific policy function for each state and each discrete choice
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
        - value_container (np.ndarray): "Filled" 3d array containing the
            choice-specific value functions for each state and each discrete choice.
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].

    """
    params_dict = convert_params_to_dict(params)

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
        compute_upper_envelope,
        transition_vector_by_state,
    ) = get_partial_functions(
        params_dict,
        options,
        user_utility_functions=utility_functions,
        user_budget_constraint=budget_constraint,
        exogenous_transition_function=transition_function,
    )

    (
        states_choice_admissible,
        indexer_states_admissible_choices,
    ) = create_state_space_admissible_choices(
        state_space, state_indexer, get_state_specific_choice_set
    )

    child_states_ids = get_child_states_index(
        state_space_admissible_choices=states_choice_admissible,
        state_indexer=state_indexer,
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

    endog_grid_container, policy_container, value_container = create_multi_dim_arrays(
        state_space, options
    )

    endog_grid_container, policy_container, value_container = backwards_induction(
        endog_grid_container=endog_grid_container,
        policy_container=policy_container,
        value_container=value_container,
        exogenous_savings_grid=exogenous_savings_grid,
        state_indexer=state_indexer,
        state_space=state_space,
        states_choice_admissible=states_choice_admissible,
        indexer_states_admissible_choices=indexer_states_admissible_choices,
        child_states_ids=child_states_ids,
        income_shock_draws=income_shock_draws,
        income_shock_weights=income_shock_weights,
        choice_set_array=choice_set_array,
        n_periods=n_periods,
        taste_shock_scale=taste_shock_scale,
        discount_factor=discount_factor,
        interest_rate=interest_rate,
        compute_marginal_utility=compute_marginal_utility,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
        compute_value=compute_value,
        compute_next_period_wealth=compute_next_period_wealth,
        get_state_specific_choice_set=get_state_specific_choice_set,
        transition_vector_by_state=transition_vector_by_state,
        compute_upper_envelope=compute_upper_envelope,
        final_period_partial=final_period_partial,
    )

    return endog_grid_container, policy_container, value_container


def backwards_induction(
    endog_grid_container: np.ndarray,
    policy_container: np.ndarray,
    value_container: np.ndarray,
    exogenous_savings_grid: np.ndarray,
    choice_set_array: np.ndarray,
    state_indexer: np.ndarray,
    state_space: np.ndarray,
    states_choice_admissible,
    indexer_states_admissible_choices,
    child_states_ids: np.ndarray,
    income_shock_draws: np.ndarray,
    income_shock_weights: np.ndarray,
    n_periods: int,
    taste_shock_scale: float,
    discount_factor: float,
    interest_rate: float,
    compute_marginal_utility: Callable,
    compute_inverse_marginal_utility: Callable,
    compute_value: Callable,
    compute_next_period_wealth: Callable,
    get_state_specific_choice_set: Callable,
    transition_vector_by_state: Callable,
    compute_upper_envelope: Callable,
    final_period_partial: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Do backwards induction and solve for optimal policy and value function.

    Args:
        endog_grid_container (np.ndarray): "Empty" 3d np.ndarray storing the
            endogenous grid for each state and each discrete choice.
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
        policy_container (np.ndarray): "Empty" 3d np.ndarray storing the
            choice-specific policy function for each state and each discrete choice
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
        value_container (np.ndarray): "Empty" 3d np.ndarray storing the
            choice-specific value functions for each state and each discrete choice.
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
        exogenous_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the exogenous savings grid.
        choice_set_array (np.ndarray): Binary array indicating if choice is
            possible.
        state_indexer (np.ndarray): Indexer object that maps states to indexes.
            The shape of this object quite complicated. For each state variable it
             has the number of possible states as "row", i.e.
            (n_poss_states_statesvar_1, n_poss_states_statesvar_2, ....)
        state_space (np.ndarray): Collection of all possible states of shape
            (n_states, n_state_variables).
        income_shock_draws (np.ndarray): 1d array of shape (n_quad_points,)
            containing the Hermite quadrature points.
        income_shock_weights (np.ndarrray): Weights for each stoachstic shock draw.
            Shape is (n_stochastic_quad_points)
        n_periods (int): Number of periods.
        taste_shock_scale (float): The taste shock scale.
        discount_factor (float): The discount factor.
        interest_rate (float): The interest rate of capital.
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled
            in.
        compute_inverse_marginal_utility (Callable): Function for calculating the
            inverse marginal utiFality, which takes the marginal utility as only
             input.
        compute_value (callable): Function for calculating the value from
            consumption level, discrete choice and expected value. The inputs
            ```discount_rate``` and ```compute_utility``` are already partialled in.
        compute_next_period_wealth (callable): User-defined function to compute the
            agent's wealth of the next period (t + 1). The inputs
            ```saving```, ```shock```, ```params``` and ```options```
            are already partialled in.
        get_state_specific_choice_set (Callable): User-supplied function returning
            for each state all possible choices.
        transition_vector_by_state (Callable): Partialled transition function return
            transition vector for each state.
        compute_upper_envelope (Callable): Function for calculating the upper
            envelope of the policy and value function. If the number of discrete
            choices is 1, this function is a dummy function that returns the policy
            and value function as is, without performing a fast upper envelope scan.
        final_period_partial (Callable): Partialled function for calculating the
            consumption as well as value function and marginal utility in the final
            period.

    Returns:
        tuple:

        - endog_grid_container (np.ndarray): "Filled" 3d array containing the
            endogenous grid for each state and each discrete choice.
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
        - policy_container (np.ndarray): "Filled" 3d array containing the
            choice-specific policy function for each state and each discrete choice
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
        - value_container (np.ndarray): "Filled" 3d array containing the
            choice-specific value functions for each state and each discrete choice.
            Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].

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
        endog_grid_container[final_state_cond, ..., : len(exogenous_savings_grid)],
        policy_container[final_state_cond, ..., : len(exogenous_savings_grid)],
        value_container[final_state_cond, ..., : len(exogenous_savings_grid)],
        marginal_utilities[final_state_cond],
        max_expected_values[final_state_cond],
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

    for period in range(n_periods - 2, -1, -1):
        periods_state_cond = np.where(state_space[:, 0] == period)[0]
        state_subspace = state_space[periods_state_cond]

        period_states_choices_cond = np.where(states_choice_admissible[:, 0] == period)[
            0
        ]
        states_choices_subset = states_choice_admissible[period_states_choices_cond]
        child_states_ids_subset = child_states_ids[period_states_choices_cond]

        marginal_utilities_child_states = np.take(
            marginal_utilities, child_states_ids_subset, axis=0
        )
        max_expected_values_child_states = np.take(
            max_expected_values, child_states_ids_subset, axis=0
        )

        (
            endog_grid_substates,
            policy_substates,
            value_substates,
            expected_value,
        ) = vmap(
            vmap(
                compute_optimal_policy_and_value,
                in_axes=(1, 1, 0, None, None, None, None, None, None),
            ),
            in_axes=(0, 0, None, None, None, None, 0, None, None),
        )(
            marginal_utilities_child_states,
            max_expected_values_child_states,
            exogenous_savings_grid,
            transition_vector_by_state,
            discount_factor,
            interest_rate,
            states_choices_subset,
            compute_inverse_marginal_utility,
            compute_value,
        )

        for id_subspace, state_choices in enumerate(states_choices_subset):
            state = state_choices[:-1]
            choice = state_choices[-1]
            current_state_index = state_indexer[tuple(state)]

            endog_grid = endog_grid_substates[id_subspace, :]
            policy = policy_substates[id_subspace, :]
            value = value_substates[id_subspace, :]

            endog_grid, policy, value = compute_upper_envelope(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                expected_value_zero_savings=expected_value[id_subspace, 0],
                exog_grid=exogenous_savings_grid,
                choice=choice,
                compute_value=compute_value,
            )

            endog_grid_container[
                current_state_index,
                choice,
                : endog_grid.shape[0],
            ] = endog_grid
            policy_container[
                current_state_index,
                choice,
                : policy.shape[0],
            ] = policy
            value_container[
                current_state_index,
                choice,
                : value.shape[0],
            ] = value

        endog_grid_child_states = endog_grid_container[periods_state_cond]
        values_child_states = value_container[periods_state_cond]
        policies_child_states = policy_container[periods_state_cond]
        choices_child_states = choice_set_array[periods_state_cond]

        (
            marginal_utilities[periods_state_cond, :],
            max_expected_values[periods_state_cond, :],
        ) = marginal_util_and_exp_max_value_states_period_jitted(
            possible_child_states=state_subspace,
            choices_child_states=choices_child_states,
            endog_grid_child_states=endog_grid_child_states,
            policies_child_states=policies_child_states,
            values_child_states=values_child_states,
        )

    return endog_grid_container, policy_container, value_container
