"""Interface for the DC-EGM algorithm."""
from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from dcegm.egm import calculate_candidate_solutions_from_euler_equation
from dcegm.final_period import save_final_period_solution
from dcegm.final_period import solve_final_period
from dcegm.integration import quadrature_legendre
from dcegm.interpolation import interpolate_and_calc_marginal_utilities
from dcegm.marg_utilities_and_exp_value import (
    aggregate_marg_utils_exp_values,
)
from dcegm.pre_processing import convert_params_to_dict
from dcegm.pre_processing import create_multi_dim_arrays
from dcegm.pre_processing import get_partial_functions
from dcegm.state_space import create_current_state_and_state_choice_objects
from dcegm.state_space import create_state_choice_space
from dcegm.state_space import get_map_from_state_to_child_nodes
from jax import vmap
from numpy.testing import assert_array_almost_equal as aaae


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

    state_space, map_state_to_state_space_index = create_state_space(options)
    (
        state_choice_space,
        map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat,
        transform_between_state_and_state_choice_space,
    ) = create_state_choice_space(
        state_space,
        map_state_to_state_space_index,
        state_space_functions["get_state_specific_choice_set"],
    )

    map_state_to_post_decision_child_nodes = get_map_from_state_to_child_nodes(
        state_space=state_space,
        state_choice_space=state_choice_space,
        map_state_to_index=map_state_to_state_space_index,
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

    backwards_induction(
        map_state_choice_vec_to_parent_state=map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat=reshape_state_choice_vec_to_mat,
        transform_between_state_and_state_choice_space=transform_between_state_and_state_choice_space,
        endog_grid_container=endog_grid_container,
        policy_container=policy_container,
        value_container=value_container,
        exogenous_savings_grid=exogenous_savings_grid,
        state_space=state_space,
        state_choice_space=state_choice_space,
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


def backwards_induction(
    map_state_choice_vec_to_parent_state: np.ndarray,
    reshape_state_choice_vec_to_mat: np.ndarray,
    transform_between_state_and_state_choice_space: np.ndarray,
    endog_grid_container: np.ndarray,
    policy_container: np.ndarray,
    value_container: np.ndarray,
    exogenous_savings_grid: np.ndarray,
    state_space: np.ndarray,
    state_choice_space,
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
    # Calculate beginning of period resources for all periods, given exogenous savings
    # and income shocks from last period
    resources_beginning_of_period = vmap(
        vmap(
            vmap(compute_next_period_wealth, in_axes=(None, None, 0)),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, None, None),
    )(state_space, exogenous_savings_grid, income_shock_draws)

    (
        idxs_state_choice_combs_final_period,
        state_choice_combs_final_period,
        endog_grid_final_period,
        reshape_current_state_choice_vec_to_mat,
        transform_between_state_and_state_choice_vec,
    ) = create_current_state_and_state_choice_objects(
        period=n_periods - 1,
        state_space=state_space,
        state_choice_space=state_choice_space,
        resources_beginning_of_period=resources_beginning_of_period,
        map_state_choice_vec_to_parent_state=map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat=reshape_state_choice_vec_to_mat,
        transform_between_state_and_state_choice_space=transform_between_state_and_state_choice_space,
    )

    (
        value_interpolated,
        policy_final_period,
        marg_util_interpolated,
    ) = solve_final_period(
        final_period_choice_states=state_choice_combs_final_period,
        final_period_solution_partial=final_period_solution_partial,
        resources_last_period=endog_grid_final_period,
    )

    (
        value_container,
        endog_grid_container,
        policy_container,
    ) = save_final_period_solution(
        endog_grid_container=endog_grid_container,
        policy_container=policy_container,
        value_container=value_container,
        idx_state_choices_final_period=idxs_state_choice_combs_final_period,
        endog_grid_final_period=endog_grid_final_period,
        policy_final_period=policy_final_period,
        value_final_period=value_interpolated,
        num_income_shock_draws=len(income_shock_draws),
        num_wealth_grid_points=len(exogenous_savings_grid),
    )

    for idx_final_period in idxs_state_choice_combs_final_period:
        # Choose which draw we take for policy and value function as those are note
        # saved with respect to the draws
        middle_of_draws = int(len(income_shock_draws) + 1 / 2)
        np.save(
            f"endog_grid_{idx_final_period}.npy",
            endog_grid_final_period[:, :, middle_of_draws],
        )
        np.save(
            f"policy_{idx_final_period}.npy", policy_final_period[:, :, middle_of_draws]
        )
        np.save(
            f"value_{idx_final_period}.npy", value_interpolated[:, :, middle_of_draws]
        )

    endog_grid_state_choice = np.empty(
        (state_choice_space.shape[1] - 1, int(1.1 * len(exogenous_savings_grid)))
    )
    policy_state_choice = np.empty(
        (state_choice_space.shape[1] - 1, int(1.1 * len(exogenous_savings_grid)))
    )
    value_state_choice = np.empty(
        (state_choice_space.shape[1] - 1, int(1.1 * len(exogenous_savings_grid)))
    )

    for period in range(n_periods - 2, -1, -1):
        endog_grid_state_choice[:] = np.nan
        policy_state_choice[:] = np.nan
        value_state_choice[:] = np.nan

        # Aggregate the marginal utilities and expected values over all choices and
        # income shock draws
        marg_util, emax = aggregate_marg_utils_exp_values(
            value_state_choice_specific=value_interpolated,
            marg_util_state_choice_specific=marg_util_interpolated,
            reshape_state_choice_vec_to_mat=reshape_current_state_choice_vec_to_mat,
            transform_between_state_and_state_choice_vec=transform_between_state_and_state_choice_vec,
            taste_shock_scale=taste_shock_scale,
            income_shock_weights=income_shock_weights,
        )

        (
            idx_state_choices_period,
            state_choices_period,
            resources_period,
            reshape_current_state_choice_vec_to_mat,
            transform_between_state_and_state_choice_vec,
        ) = create_current_state_and_state_choice_objects(
            period=period,
            state_space=state_space,
            state_choice_space=state_choice_space,
            resources_beginning_of_period=resources_beginning_of_period,
            map_state_choice_vec_to_parent_state=map_state_choice_vec_to_parent_state,
            reshape_state_choice_vec_to_mat=reshape_state_choice_vec_to_mat,
            transform_between_state_and_state_choice_space=transform_between_state_and_state_choice_space,
        )

        (
            endog_grid_candidate,
            value_candidate,
            policy_candidate,
            expected_values,
        ) = calculate_candidate_solutions_from_euler_equation(
            marg_util=marg_util,
            emax=emax,
            idx_state_choices_period=idx_state_choices_period,
            map_state_to_post_decision_child_nodes=map_state_to_post_decision_child_nodes,
            exogenous_savings_grid=exogenous_savings_grid,
            transition_vector_by_state=transition_vector_by_state,
            discount_factor=discount_factor,
            interest_rate=interest_rate,
            state_choices_period=state_choices_period,
            compute_inverse_marginal_utility=compute_inverse_marginal_utility,
            compute_value=compute_value,
        )

        # Run upper envolope to remove suboptimal candidates
        for state_choice_idx, state_choice_vec in enumerate(state_choices_period):
            _idx = 0
            choice = state_choice_vec[-1]

            endog_grid, policy, value = compute_upper_envelope(
                endog_grid=endog_grid_candidate[state_choice_idx],
                policy=policy_candidate[state_choice_idx],
                value=value_candidate[state_choice_idx],
                expected_value_zero_savings=expected_values[state_choice_idx, 0],
                exog_grid=exogenous_savings_grid,
                choice=choice,
                compute_value=compute_value,
            )

            _idx_state_choice_full = idx_state_choices_period[state_choice_idx]

            np.save(f"endog_grid_{_idx_state_choice_full}.npy", endog_grid)
            np.save(f"policy_{_idx_state_choice_full}.npy", policy)
            np.save(f"value_{_idx_state_choice_full}.npy", value)

            endog_grid_container[_idx_state_choice_full, : len(endog_grid)] = endog_grid
            policy_container[_idx_state_choice_full, : len(policy)] = policy
            value_container[_idx_state_choice_full, : len(value)] = value

            endog_grid_state_choice[_idx, : len(endog_grid)] = endog_grid
            policy_state_choice[_idx, : len(policy)] = policy
            value_state_choice[_idx, : len(value)] = value

            aaae(
                endog_grid_container[_idx_state_choice_full, : len(endog_grid)],
                endog_grid_state_choice[_idx, : len(endog_grid)],
            )
            aaae(
                policy_container[_idx_state_choice_full, : len(policy)],
                policy_state_choice[_idx, : len(policy)],
            )
            aaae(
                value_container[_idx_state_choice_full, : len(value)],
                value_state_choice[_idx, : len(value)],
            )
            _idx += 1

        marg_util_interpolated, value_interpolated = vmap(
            interpolate_and_calc_marginal_utilities, in_axes=(None, None, 0, 0, 0, 0, 0)
        )(
            compute_marginal_utility,
            compute_value,
            state_choices_period[:, -1],
            resources_period,
            endog_grid_container[idx_state_choices_period, :],
            policy_container[idx_state_choices_period, :],
            value_container[idx_state_choices_period, :],
            # endog_grid_state_choice_current,
            # policy_state_choice_current,
            # value_state_choice_current,
        )
