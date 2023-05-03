"""Interface for the DC-EGM algorithm."""
from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import jax.numpy as jnp
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
from dcegm.state_space import create_state_choice_space
from dcegm.state_space import get_child_states_index
from dcegm.state_space import get_feasible_choice_space
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
    taste_shock = params_dict["lambda"]
    interest_rate = params_dict["interest_rate"]
    discount_factor = params_dict["beta"]
    max_wealth = params_dict["max_wealth"]

    n_periods = options["n_periods"]
    n_grid_wealth = options["grid_points_wealth"]
    exogenous_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

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
    create_state_space = state_space_functions["create_state_space"]
    get_state_specific_choice_set = state_space_functions[
        "get_state_specific_choice_set"
    ]

    state_space, state_indexer = create_state_space(options)
    state_choice_space, indexer_state_choice_space = create_state_choice_space(
        state_space, state_indexer, get_state_specific_choice_set
    )

    child_states_ids = get_child_states_index(
        state_choice_space=state_choice_space,
        map_state_to_index=state_indexer,
    )

    final_period_partial = partial(
        final_period_wrapper,
        options=options,
        compute_utility=compute_utility,
        final_period_solution=final_period_solution,
    )

    binary_choice_space = get_feasible_choice_space(
        state_space=state_space,
        map_states_to_indices=state_indexer,
        get_state_specific_choice_set=get_state_specific_choice_set,
        options=options,
    )

    endog_grid_container, policy_container, value_container = create_multi_dim_arrays(
        state_space, options
    )

    endog_grid_container, policy_container, value_container = backwards_induction(
        endog_grid_container=endog_grid_container,
        policy_container=policy_container,
        value_container=value_container,
        exogenous_savings_grid=exogenous_savings_grid,
        map_state_to_index=state_indexer,
        state_space=state_space,
        state_choice_space=state_choice_space,
        indexer_state_choice_space=indexer_state_choice_space,
        map_current_state_to_child_nodes=child_states_ids,
        income_shock_draws=income_shock_draws,
        income_shock_weights=income_shock_weights,
        binary_choice_space=binary_choice_space,
        n_periods=n_periods,
        taste_shock=taste_shock,
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
    binary_choice_space: np.ndarray,
    map_state_to_index: np.ndarray,
    state_space: np.ndarray,
    state_choice_space,
    indexer_state_choice_space,
    map_current_state_to_child_nodes: np.ndarray,
    income_shock_draws: np.ndarray,
    income_shock_weights: np.ndarray,
    n_periods: int,
    taste_shock: float,
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
    get_marg_util_and_emax_jitted = jit(
        partial(
            marginal_util_and_exp_max_value_states_period,
            compute_next_period_wealth=compute_next_period_wealth,
            compute_marginal_utility=compute_marginal_utility,
            compute_value=compute_value,
            taste_shock_scale=taste_shock,
            exogenous_savings_grid=exogenous_savings_grid,
            income_shock_draws=income_shock_draws,
            income_shock_weights=income_shock_weights,
        )
    )

    idx_state_space_final = np.where(state_space[:, 0] == n_periods - 1)[0]
    states_final = state_space[idx_state_space_final]

    (
        endog_grid_container[idx_state_space_final, ..., : len(exogenous_savings_grid)],
        policy_container[idx_state_space_final, ..., : len(exogenous_savings_grid)],
        value_container[idx_state_space_final, ..., : len(exogenous_savings_grid)],
        _marg_utils_next,
        _emax_next,
    ) = final_period_partial(
        final_period_states=states_final,
        choices_final=binary_choice_space[idx_state_space_final],
        compute_next_period_wealth=compute_next_period_wealth,
        compute_marginal_utility=compute_marginal_utility,
        taste_shock=taste_shock,
        exogenous_savings_grid=exogenous_savings_grid,
        income_shock_draws=income_shock_draws,
        income_shock_weights=income_shock_weights,
    )

    for period in range(n_periods - 2, -1, -1):
        idx_state_space_current = np.where(state_space[:, 0] == period)[0]
        idx_state_choice_space_current = np.where(state_choice_space[:, 0] == period)[0]

        state_space_current = state_space[idx_state_space_current]
        state_choice_space_current = state_choice_space[idx_state_choice_space_current]

        _idx_child_nodes = map_current_state_to_child_nodes[
            idx_state_choice_space_current
        ]
        idx_child_nodes = _idx_child_nodes - (period + 1) * (
            state_space.shape[0] // n_periods
        )
        marg_utils_next = jnp.take(_marg_utils_next, idx_child_nodes, axis=0)
        emax_next = jnp.take(_emax_next, idx_child_nodes, axis=0)

        (
            endog_grid_substates,
            policy_substates,
            value_substates,
            expected_value_substates,
        ) = vmap(
            vmap(
                compute_optimal_policy_and_value,
                in_axes=(1, 1, 0, None, None, None, None, None, None),
            ),
            in_axes=(0, 0, None, None, None, None, 0, None, None),
        )(
            marg_utils_next,
            emax_next,
            exogenous_savings_grid,
            transition_vector_by_state,
            discount_factor,
            interest_rate,
            state_choice_space_current,
            compute_inverse_marginal_utility,
            compute_value,
        )

        for idx, state_choice_vec in enumerate(state_choice_space_current):
            state_vec = state_choice_vec[:-1]
            choice = state_choice_vec[-1]

            idx_state = map_state_to_index[tuple(state_vec)]  #

            endog_grid = endog_grid_substates[idx, :]
            policy = policy_substates[idx, :]
            value = value_substates[idx, :]

            endog_grid, policy, value = compute_upper_envelope(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                expected_value_zero_savings=expected_value_substates[idx, 0],
                exog_grid=exogenous_savings_grid,
                choice=choice,
                compute_value=compute_value,
            )

            endog_grid_container[idx_state, choice, : len(endog_grid)] = endog_grid
            policy_container[idx_state, choice, : len(policy)] = policy
            value_container[idx_state, choice, : len(value)] = value

        endog_grid_child_states = endog_grid_container[idx_state_space_current]
        values_child_states = value_container[idx_state_space_current]
        policies_child_states = policy_container[idx_state_space_current]

        choices_child_states = binary_choice_space[idx_state_space_current]

        _marg_utils_next, _emax_next = get_marg_util_and_emax_jitted(
            state_space_next=state_space_current,
            choices_child_states=choices_child_states,
            endog_grid_child_states=endog_grid_child_states,
            policies_child_states=policies_child_states,
            values_child_states=values_child_states,
        )

    return endog_grid_container, policy_container, value_container
