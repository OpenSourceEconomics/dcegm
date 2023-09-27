"""Interface for the DC-EGM algorithm."""
from functools import partial
from typing import Callable
from typing import Dict

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.egm import calculate_candidate_solutions_from_euler_equation
from dcegm.integration import quadrature_legendre
from dcegm.interpolation import interpolate_and_calc_marginal_utilities
from dcegm.marg_utilities_and_exp_value import (
    aggregate_marg_utils_exp_values,
)
from dcegm.process_model import convert_params_to_dict
from dcegm.process_model import process_model_functions
from dcegm.state_space import create_map_from_state_to_child_nodes
from dcegm.state_space import (
    create_period_state_and_state_choice_objects,
)
from dcegm.state_space import create_state_choice_space
from jax import jit
from jax import vmap


def get_solve_function(
    options: Dict[str, int],
    exog_savings_grid: jnp.ndarray,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    final_period_solution: Callable,
    # transition_function: Callable,
) -> Callable:
    """Create a solve function, which only takes params as input.

    Args:
        options (dict): Options dictionary.
        exog_savings_grid (jnp.ndarray): 1d array of shape (n_grid_wealth,) containing
            the user-supplied exogenous savings grid.
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of:
            (i) utility
            (ii) inverse marginal utility
            (iii) next period marginal utility
        budget_constraint (callable): Callable budget constraint.
        state_space_functions (Dict[str, callable]): Dictionary of two user-supplied
            functions to:
            (i) create the state space
            (ii) get the state specific feasible choice set
            (iii) update the endogenous part of the state by the choice
        final_period_solution (callable): User-supplied function for solving the agent's
            last period.
        transition_function (callable): User-supplied function returning for each
            state a transition matrix vector.

    Returns:
        callable: The partial solve function that only takes ```params``` as input.

    """
    model_params = options["model_params"]

    n_periods = model_params["n_periods"]

    # ToDo: Make interface with several draw possibilities.
    # ToDo: Some day make user supplied draw function.
    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        model_params["quadrature_points_stochastic"]
    )

    create_state_space = state_space_functions["create_state_space"]
    state_space, map_state_to_state_space_index = create_state_space(model_params)
    (
        state_choice_space,
        map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat,
    ) = create_state_choice_space(
        model_params,
        state_space,
        map_state_to_state_space_index,
        state_space_functions["get_state_specific_choice_set"],
    )

    #
    period_specific_state_objects = create_period_state_and_state_choice_objects(
        options=model_params,
        state_space=state_space,
        state_choice_space=state_choice_space,
        map_state_choice_vec_to_parent_state=map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat=reshape_state_choice_vec_to_mat,
    )

    period_specific_state_objects = create_map_from_state_to_child_nodes(
        options=model_params,
        period_specific_state_objects=period_specific_state_objects,
        map_state_to_index=map_state_to_state_space_index,
        update_endog_state_by_state_and_choice=state_space_functions[
            "update_endog_state_by_state_and_choice"
        ],
    )
    #

    (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_beginning_of_period_wealth,
        compute_final_period,
        compute_exog_transition_vec,
        compute_upper_envelope,
    ) = process_model_functions(
        options,
        user_utility_functions=utility_functions,
        user_budget_constraint=budget_constraint,
        user_final_period_solution=final_period_solution,
    )

    backward_jit = jit(
        partial(
            backward_induction,
            period_specific_state_objects=period_specific_state_objects,
            exog_savings_grid=exog_savings_grid,
            state_space=state_space,
            income_shock_draws_unscaled=income_shock_draws_unscaled,
            income_shock_weights=income_shock_weights,
            n_periods=n_periods,
            compute_utility=compute_utility,
            compute_marginal_utility=compute_marginal_utility,
            compute_inverse_marginal_utility=compute_inverse_marginal_utility,
            compute_beginning_of_period_wealth=compute_beginning_of_period_wealth,
            compute_final_period=compute_final_period,
            compute_exog_transition_vec=compute_exog_transition_vec,
            compute_upper_envelope=compute_upper_envelope,
        )
    )

    def solve_func(params):
        params_initial = convert_params_to_dict(params)
        return backward_jit(params=params_initial)

    return solve_func


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict[str, int],
    exog_savings_grid: jnp.ndarray,
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable],
    final_period_solution: Callable,
) -> Dict[int, np.ndarray]:
    """Solve a discrete-continuous life-cycle model using the DC-EGM algorithm.

    Args:
        params (pd.DataFrame): Params DataFrame.
        options (dict): Options dictionary.
        exog_savings_grid (jnp.ndarray): 1d array of shape (n_grid_wealth,) containing
            the user-supplied exogenous savings grid.
        utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of:
            (i) utility
            (ii) inverse marginal utility
            (iii) next period marginal utility
        budget_constraint (callable): Callable budget constraint.
        state_space_functions (Dict[str, callable]): Dictionary of two user-supplied
            functions to:
            (i) create the state space
            (ii) get the state specific feasible choice set
            (iii) update the endogenous part of the state by the choice
        final_period_solution (callable): User-supplied function for solving the agent's
            last period.
        transition_function (callable): User-supplied function returning for each
            state a transition matrix vector.

    Returns:
        dict: Dictionary containing the period-specific endog_grid, policy_left,
            policy_right, and value from the backward induction.

    """
    backward_jit = get_solve_function(
        options=options,
        exog_savings_grid=exog_savings_grid,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        budget_constraint=budget_constraint,
        final_period_solution=final_period_solution,
    )

    results = backward_jit(params=params)

    return results


def backward_induction(
    params: Dict[str, float],
    period_specific_state_objects: Dict[int, jnp.ndarray],
    exog_savings_grid: np.ndarray,
    state_space: np.ndarray,
    income_shock_draws_unscaled: np.ndarray,
    income_shock_weights: np.ndarray,
    n_periods: int,
    compute_utility: Callable,
    compute_marginal_utility: Callable,
    compute_inverse_marginal_utility: Callable,
    compute_beginning_of_period_wealth: Callable,
    compute_final_period: Callable,
    compute_exog_transition_vec: Callable,
    compute_upper_envelope: Callable,
) -> Dict[int, np.ndarray]:
    """Do backward induction and solve for optimal policy and value function.

    Args:
        params (dict): Dictionary containing the model parameters.
        options (dict): Options dictionary.
        period_specific_state_objects (np.ndarray): Dictionary containing
            period-specific state and state-choice objects, with the following keys:
            - "state_choice_mat" (jnp.ndarray)
            - "idx_state_of_state_choice" (jnp.ndarray)
            - "reshape_state_choice_vec_to_mat" (callable)
            - "transform_between_state_and_state_choice_vec" (callable)
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
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled
            in.
        compute_inverse_marginal_utility (Callable): Function for calculating the
            inverse marginal utiFality, which takes the marginal utility as only
             input.
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
        dict: Dictionary containing the period-specific endog_grid, policy_left,
            policy_right, and value from the backward induction.

    """

    taste_shock_scale = params["lambda"]
    income_shock_draws = income_shock_draws_unscaled * params["sigma"]

    state_objects = period_specific_state_objects[n_periods - 1]

    resources_beginning_of_period = vmap(
        vmap(
            vmap(
                compute_beginning_of_period_wealth,
                in_axes=(None, None, 0, None),
            ),
            in_axes=(None, 0, None, None),
        ),
        in_axes=(0, None, None, None),
    )(state_space, exog_savings_grid, income_shock_draws, params)

    resources_final_period = resources_beginning_of_period[
        state_objects["idx_parent_states"]
    ]

    marg_util_interpolated, value_interpolated, policy_final = vmap(
        vmap(
            vmap(
                compute_final_period,
                in_axes=(None, 0, None),
            ),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, 0, None),
    )(
        state_objects["state_choice_mat"],
        resources_beginning_of_period[state_objects["idx_parent_states"]],
        params,
    )

    # Choose which draw we take for policy and value function as those are not
    # saved with respect to the draws
    middle_of_draws = int(len(income_shock_draws) + 1 / 2)

    final_period_results = {}
    final_period_results["value"] = value_interpolated[:, :, middle_of_draws]
    final_period_results["policy_left"] = policy_final[:, :, middle_of_draws]
    final_period_results["policy_right"] = policy_final[:, :, middle_of_draws]
    final_period_results["endog_grid"] = resources_final_period[:, :, middle_of_draws]

    results = {}
    results[n_periods - 1] = final_period_results

    for period in range(n_periods - 2, -1, -1):
        state_objects = period_specific_state_objects[period]

        # Aggregate the marginal utilities and expected values over all choices and
        # income shock draws
        marg_util, emax = aggregate_marg_utils_exp_values(
            value_state_choice_specific=value_interpolated,
            marg_util_state_choice_specific=marg_util_interpolated,
            reshape_state_choice_vec_to_mat=state_objects[
                "reshape_state_choice_vec_to_mat"
            ],
            taste_shock_scale=taste_shock_scale,
            income_shock_weights=income_shock_weights,
        )

        (
            endog_grid_candidate,
            value_candidate,
            policy_candidate,
            expected_values,
        ) = calculate_candidate_solutions_from_euler_equation(
            exogenous_savings_grid=exog_savings_grid,
            marg_util=marg_util,
            emax=emax,
            state_choice_vec=state_objects["state_choice_mat"],  # state_vec and choice
            # transition_vec=transition_mat_dict[period],
            idx_post_decision_child_states=state_objects["idx_feasible_child_nodes"],
            compute_inverse_marginal_utility=compute_inverse_marginal_utility,
            compute_utility=compute_utility,
            compute_exog_transition_vec=compute_exog_transition_vec,
            params=params,
        )

        # Run upper envelope to remove suboptimal candidates
        (
            endog_grid_state_choice,
            policy_left_state_choice,
            policy_right_state_choice,
            value_state_choice,
        ) = vmap(
            compute_upper_envelope,
            in_axes=(0, 0, 0, 0, 0, None, None),  # vmap over state-choice combs
        )(
            endog_grid_candidate,
            policy_candidate,
            value_candidate,
            expected_values[:, 0],
            state_objects["state_choice_mat"],  # state_vec and choice
            params,
            compute_utility,
        )
        resources_period = resources_beginning_of_period[
            state_objects["idx_parent_states"]
        ]

        # ToDo: reorder function arguments
        marg_util_interpolated, value_interpolated = vmap(
            interpolate_and_calc_marginal_utilities,
            in_axes=(None, None, 0, 0, 0, 0, 0, 0, None),
        )(
            compute_marginal_utility,
            compute_utility,
            state_objects["state_choice_mat"],  # state_vec and choice
            # state_objects["state_choice_mat"][:, -1],  # choice
            resources_period,
            endog_grid_state_choice,
            policy_left_state_choice,
            policy_right_state_choice,
            value_state_choice,
            params,
        )

        period_results = {}
        period_results["policy_left"] = policy_left_state_choice
        period_results["policy_right"] = policy_right_state_choice
        period_results["endog_grid"] = endog_grid_state_choice
        period_results["value"] = value_state_choice

        results[period] = period_results

    return results
