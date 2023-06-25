"""Interface for the DC-EGM algorithm."""
from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.egm import compute_optimal_policy_and_value
from dcegm.final_period import aggregate_marg_utils_exp_values
from dcegm.final_period import final_period_wrapper
from dcegm.integration import quadrature_legendre
from dcegm.marg_utilities_and_exp_value import (
    marginal_util_and_exp_max_value_states_period,
)
from dcegm.pre_processing import convert_params_to_dict
from dcegm.pre_processing import create_multi_dim_arrays
from dcegm.pre_processing import get_partial_functions
from dcegm.state_space import create_state_choice_space
from dcegm.state_space import get_map_from_state_to_child_nodes
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

    state_space, map_state_to_index = create_state_space(options)
    (
        state_choice_space,
        sum_state_choices_to_state,
        map_state_choice_to_state,
        state_times_state_choice_mat,
    ) = create_state_choice_space(
        state_space,
        map_state_to_index,
        state_space_functions["get_state_specific_choice_set"],
    )

    map_state_to_post_decision_child_nodes = get_map_from_state_to_child_nodes(
        state_space=state_space,
        state_choice_space=state_choice_space,
        map_state_to_index=map_state_to_index,
    )

    final_period_solution_partial = partial(
        final_period_solution,
        params_dict=params_dict,
        options=options,
        compute_utility=compute_utility,
        compute_marginal_utility=compute_marginal_utility,
    )

    endog_grid_container, policy_container, value_container = create_multi_dim_arrays(
        state_choice_space, options
    )

    endog_grid_container, policy_container, value_container = backwards_induction(
        sum_state_choices_to_state,
        map_state_choice_to_state,
        state_times_state_choice_mat,
        int(options["n_discrete_choices"]),
        endog_grid_container=endog_grid_container,
        policy_container=policy_container,
        value_container=value_container,
        exogenous_savings_grid=exogenous_savings_grid,
        state_space=state_space,
        state_choice_space=state_choice_space,
        map_state_to_index=map_state_to_index,
        map_state_to_post_decision_child_nodes=map_state_to_post_decision_child_nodes,
        income_shock_draws=income_shock_draws,
        income_shock_weights=income_shock_weights,
        n_periods=n_periods,
        taste_shock_scale=taste_shock_scale,
        discount_factor=discount_factor,
        interest_rate=interest_rate,
        compute_marginal_utility=compute_marginal_utility,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
        compute_value=compute_value,
        compute_next_period_wealth=compute_next_period_wealth,
        transition_vector_by_state=transition_vector_by_state,
        compute_upper_envelope=compute_upper_envelope,
        final_period_solution_partial=final_period_solution_partial,
    )

    # ToDo: finalize output containers

    return endog_grid_container, policy_container, value_container


def backwards_induction(
    sum_state_choices_to_state,
    map_state_choice_to_state,
    state_times_state_choice_mat: np.ndarray,
    n_choices,
    endog_grid_container: np.ndarray,
    policy_container: np.ndarray,
    value_container: np.ndarray,
    exogenous_savings_grid: np.ndarray,
    state_space: np.ndarray,
    state_choice_space,
    map_state_to_index: np.ndarray,
    map_state_to_post_decision_child_nodes: np.ndarray,
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
    transition_vector_by_state: Callable,
    compute_upper_envelope: Callable,
    final_period_solution_partial: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Do backwards induction and solve for optimal policy and value function.

        Args:
            endog_grid_container (np.ndarray): "Empty" 3d np.ndarray storing the
                endogenous grid for             compute_next_period_wealth=compute_next_period_wealth,
    each state and each discrete choice.
                Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
            policy_container (np.ndarray): "Empty" 3d np.ndarray storing the
                choice-specific policy function for each state and each discrete choice
                Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
            value_container (np.ndarray): "Empty" 3d np.ndarray storing the
                choice-specific value functions for each state and each discrete choice.
                Has shape [n_states, n_discrete_choices, 1.1 * n_grid_wealth].
            exogenous_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
                containing the exogenous savings grid.
            state_space (np.ndarray): 2d array of shape (n_states, n_state_variables + 1)
                which serves as a collection of all possible states. By convention,
                the first column must contain the period and the last column the
                exogenous processes. Any other state variables are in between.
                E.g. if the two state variables are period and lagged choice and all choices
                are admissible in each period, the shape of the state space array is
                (n_periods * n_choices, 3).
            state_choice_space (np.ndarray): 2d array of shape
                (n_feasible_states, n_state_and_exog_variables + 1) containing all
                feasible state-choice combinations. By convention, the second to last
                column contains the exogenous process. The last column always contains the
                choice to be made (which is not a state variable).
            map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
                The shape of this object is quite complicated. For each state variable it
                has the number of possible states as rows, i.e.
                (n_poss_states_state_var_1, n_poss_states_state_var_2, ....).
            map_state_to_post_decision_child_nodes (np.ndarray): 2d array of shape
                (n_feasible_state_choice_combs, n_choices * n_exog_processes)
                containing indices of all child nodes the agent can reach
                from any given state.
            income_shock_draws (np.ndarray): 1d array of shape (n_quad_points,)
                containing the Hermite quadrature points.
            income_shock_weights (np.ndarrray): 1d array of shape
                (n_stochastic_quad_points) with weights for each stoachstic shock draw.
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
            compute_marginal_utility=compute_marginal_utility,
            compute_value=compute_value,
            taste_shock_scale=taste_shock_scale,
            income_shock_weights=income_shock_weights,
        )
    )
    n_states_over_periods = state_space.shape[0] // n_periods

    resources_beginning_of_period = vmap(
        vmap(
            vmap(compute_next_period_wealth, in_axes=(None, None, 0)),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, None, None),
    )(state_space, exogenous_savings_grid, income_shock_draws)

    idx_states_final_period = np.where(state_space[:, 0] == n_periods - 1)[0]
    idx_state_choices_final_period = np.where(
        state_choice_space[:, 0] == n_periods - 1
    )[0]
    map_final_state_choice_to_state = map_state_choice_to_state[
        idx_state_choices_final_period
    ]
    last_period_sum_state_choices_to_state = sum_state_choices_to_state[
        idx_states_final_period, :
    ][:, idx_state_choices_final_period]

    resources_last_period = resources_beginning_of_period[
        map_final_state_choice_to_state
    ]
    final_period_state_choices = state_choice_space[idx_state_choices_final_period]

    boolean_choice_mat_child_states = np.full((n_states_over_periods, n_choices), False)
    for state_choice_idx, state_choice_vec in enumerate(final_period_state_choices):
        boolean_choice_mat_child_states[
            state_choice_idx - n_choices, state_choice_vec[-1]
        ] = True

    (
        endog_grid_final_period,
        policy_final_period,
        value_final_period,
        marginal_utilities_choices,
    ) = final_period_wrapper(
        final_period_choice_states=final_period_state_choices,
        state_times_state_choice_mat=state_times_state_choice_mat[
            idx_states_final_period, :
        ],
        final_period_solution_partial=final_period_solution_partial,
        resources_last_period=resources_last_period,
        sum_state_choices_to_state=last_period_sum_state_choices_to_state,
        taste_shock_scale=taste_shock_scale,
        income_shock_draws=income_shock_draws,
        income_shock_weights=income_shock_weights,
    )
    # Choose which draw we take for policy and value function as those are note saved
    # with respect to the draws
    middle_of_draws = int(income_shock_draws.shape[0] + 1 / 2)
    value_container[
        idx_state_choices_final_period, ..., : len(exogenous_savings_grid)
    ] = value_final_period[:, :, middle_of_draws]
    endog_grid_container[
        idx_state_choices_final_period, ..., : len(exogenous_savings_grid)
    ] = endog_grid_final_period[:, :, middle_of_draws]
    policy_container[
        idx_state_choices_final_period, ..., : len(exogenous_savings_grid)
    ] = policy_final_period[:, :, middle_of_draws]

    # Aggregate the marginal utilities and expected values over all choices
    marg_util, emax = aggregate_marg_utils_exp_values(
        final_value_state_choice=value_final_period,
        state_times_state_choice_mat=state_times_state_choice_mat[
            idx_states_final_period, :
        ],
        marg_util_state_choice=marginal_utilities_choices,
        sum_state_choices_to_state=last_period_sum_state_choices_to_state,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
    )

    for period in range(n_periods - 2, -1, -1):
        idx_possible_states = np.where(state_space[:, 0] == period)[0]
        idx_state_choices_period = np.where(state_choice_space[:, 0] == period)[0]
        feasible_state_choice_combs = state_choice_space[idx_state_choices_period]

        # If we are currently at period t, the child state that arises
        # from a particular decision made in period t is actually the beginning
        # of period state that arises in period t+1.
        feasible_marg_utils, feasible_emax = _get_post_decision_marg_utils_and_emax(
            marg_util_next=marg_util,
            emax_next=emax,
            idx_state_choice_combs=idx_state_choices_period,
            map_state_to_post_decision_child_nodes=map_state_to_post_decision_child_nodes,
        )

        (
            feasible_endog_grids,
            feasible_policies,
            feasible_values,
            feasible_expected_values,
        ) = vmap(
            vmap(
                compute_optimal_policy_and_value,
                in_axes=(1, 1, 0, None, None, None, None, None, None),  # savings grid
            ),
            in_axes=(0, 0, None, None, None, None, 0, None, None),  # states and choices
        )(
            feasible_marg_utils,
            feasible_emax,
            exogenous_savings_grid,
            transition_vector_by_state,
            discount_factor,
            interest_rate,
            feasible_state_choice_combs,
            compute_inverse_marginal_utility,
            compute_value,
        )

        temp_policy = np.full(
            shape=(len(idx_possible_states), n_choices, endog_grid_container.shape[-1]),
            fill_value=np.nan,
        )
        temp_value = np.full(
            shape=(len(idx_possible_states), n_choices, endog_grid_container.shape[-1]),
            fill_value=np.nan,
        )
        temp_endog_grid = np.full(
            shape=(len(idx_possible_states), n_choices, endog_grid_container.shape[-1]),
            fill_value=np.nan,
        )
        boolean_choice_mat_child_states[:] = False

        for state_choice_idx, state_choice_vec in enumerate(
            feasible_state_choice_combs
        ):
            choice = state_choice_vec[-1]
            state = state_choice_vec[:-1]

            endog_grid, policy, value = compute_upper_envelope(
                endog_grid=feasible_endog_grids[state_choice_idx],
                policy=feasible_policies[state_choice_idx],
                value=feasible_values[state_choice_idx],
                expected_value_zero_savings=feasible_expected_values[
                    state_choice_idx, 0
                ],
                exog_grid=exogenous_savings_grid,
                choice=choice,
                compute_value=compute_value,
            )
            idx_temp_state = np.where(
                map_state_to_index[tuple(state)] == idx_possible_states
            )[0]
            temp_policy[idx_temp_state, choice, : len(policy)] = policy
            temp_value[idx_temp_state, choice, : len(value)] = value
            temp_endog_grid[idx_temp_state, choice, : len(endog_grid)] = endog_grid

            _idx_child_state = state_choice_idx - n_choices
            boolean_choice_mat_child_states[_idx_child_state, choice] = True

            _idx_state_choice_full = idx_state_choices_period[state_choice_idx]
            endog_grid_container[_idx_state_choice_full, : len(endog_grid)] = endog_grid
            policy_container[_idx_state_choice_full, : len(policy)] = policy
            value_container[_idx_state_choice_full, : len(value)] = value

        marg_util, emax = get_marg_util_and_emax_jitted(
            resources_next_period=resources_beginning_of_period[idx_possible_states],
            choices_child_states=boolean_choice_mat_child_states,
            endog_grid_child_states=temp_endog_grid,
            policy_child_states=temp_policy,
            value_child_states=temp_value,
        )

        # ToDo: save arrays to disc

    # ToDo: return None
    return endog_grid_container, policy_container, value_container


def _get_post_decision_marg_utils_and_emax(
    marg_util_next,
    emax_next,
    idx_state_choice_combs,
    map_state_to_post_decision_child_nodes,
):
    """Get marginal utility and expected maximum value of post-decision child states.

    Args:
        marg_util_next (np.ndarray): 2d array of shape (n_choices, n_grid_wealth)
            containing the choice-specific marginal utilities of the next period,
            i.e. t + 1.
        emax_next (np.ndarray): 2d array of shape (n_choices, n_grid_wealth)
            containing the choice-specific expected maximum values of the next period,
            i.e. t + 1.
        idx_state_choice_combs (np.ndarray): Indexer for the state choice combinations
            that are feasible in the current period.
        map_state_to_post_decision_child_nodes (np.ndarray): Indexer for the child nodes
            that can be reached from the current state.

    Returns:
        tuple:

        - marg_utils_child (np.ndarray): 3d array of shape
            (n_child_states, n_exog_processes, n_grid_wealth) containing the
            state-choice specific marginal utilities of the child states in
            the current period t.
        - emax_child (np.ndarray): 3d array of shape
            (n_child_states, n_exog_processes, n_grid_wealth) containing the
            state-choice specific expected maximum values of the child states
            in the current period t.

    """
    idx_post_decision_child_states = map_state_to_post_decision_child_nodes[
        idx_state_choice_combs
    ]

    # state-choice specific
    marg_utils_child = jnp.take(marg_util_next, idx_post_decision_child_states, axis=0)
    emax_child = jnp.take(emax_next, idx_post_decision_child_states, axis=0)

    return marg_utils_child, emax_child
