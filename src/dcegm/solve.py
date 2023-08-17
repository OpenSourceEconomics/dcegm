"""Interface for the DC-EGM algorithm."""
from functools import partial
from typing import Callable
from typing import Dict

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.egm import calculate_candidate_solutions_from_euler_equation
from dcegm.final_period import solve_final_period
from dcegm.integration import quadrature_legendre
from dcegm.interpolation import interpolate_and_calc_marginal_utilities
from dcegm.marg_utilities_and_exp_value import (
    aggregate_marg_utils_exp_values,
)
from dcegm.pre_processing import convert_params_to_dict
from dcegm.pre_processing import get_partial_functions
from dcegm.state_space import (
    create_period_state_and_state_choice_objects,
)
from dcegm.state_space import create_state_choice_space
from dcegm.state_space import get_map_from_state_to_child_nodes
from jax import vmap


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict[str, int],
    exog_savings_grid: jnp.ndarray,
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable],
    final_period_solution: Callable,
    transition_function: Callable,
) -> None:
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
            (ii) get the state specific choice set
        final_period_solution (callable): User-supplied function for solving the agent's
            last period.
        transition_function (callable): User-supplied function returning for each
            state a transition matrix vector.

    Returns:
        None

    """
    params_dict = convert_params_to_dict(params)

    taste_shock_scale = params_dict["lambda"]
    params_dict["interest_rate"]
    params_dict["beta"]

    n_periods = options["n_periods"]
    # max_wealth = params_dict["max_wealth"]
    # n_grid_wealth = options["grid_points_wealth"]
    # exog_savings_grid = jnp.linspace(0, max_wealth, n_grid_wealth)

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
        compute_utility=compute_utility,
        compute_marginal_utility=compute_marginal_utility,
        options=options,
    )

    period_specific_state_objects = create_period_state_and_state_choice_objects(
        state_space=state_space,
        state_choice_space=state_choice_space,
        map_state_choice_vec_to_parent_state=map_state_choice_vec_to_parent_state,
        reshape_state_choice_vec_to_mat=reshape_state_choice_vec_to_mat,
        transform_between_state_and_state_choice_space=transform_between_state_and_state_choice_space,
        num_periods=n_periods,
    )

    backwards_induction(
        params=params_dict,
        period_specific_state_objects=period_specific_state_objects,
        exog_savings_grid=exog_savings_grid,
        state_space=state_space,
        map_state_to_post_decision_child_nodes=map_state_to_post_decision_child_nodes,
        income_shock_draws=income_shock_draws,
        income_shock_weights=income_shock_weights,
        n_periods=n_periods,
        taste_shock_scale=taste_shock_scale,
        compute_marginal_utility=compute_marginal_utility,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
        compute_value=compute_value,
        compute_next_period_wealth=compute_next_period_wealth,
        transition_vector_by_state=transition_vector_by_state,
        compute_upper_envelope=compute_upper_envelope,
        final_period_solution_partial=final_period_solution_partial,
    )


def backwards_induction(
    params: Dict[str, float],
    period_specific_state_objects: dict,
    exog_savings_grid: np.ndarray,
    state_space: np.ndarray,
    map_state_to_post_decision_child_nodes: np.ndarray,
    income_shock_draws: np.ndarray,
    income_shock_weights: np.ndarray,
    n_periods: int,
    taste_shock_scale: float,
    compute_marginal_utility: Callable,
    compute_inverse_marginal_utility: Callable,
    compute_value: Callable,
    compute_next_period_wealth: Callable,
    transition_vector_by_state: Callable,
    compute_upper_envelope: Callable,
    final_period_solution_partial: Callable,
) -> None:
    """Do backwards induction and solve for optimal policy and value function.

    Args:
        params_dict (dict): Dictionary of parameters.
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
        None

    """

    # Calculate beginning of period resources for all periods, given exogenous savings
    # and income shocks from last period
    begin_of_period_resources = vmap(
        vmap(
            vmap(compute_next_period_wealth, in_axes=(None, None, 0, None)),
            in_axes=(None, 0, None, None),
        ),
        in_axes=(0, None, None, None),
    )(state_space, exog_savings_grid, income_shock_draws, params)

    state_objects_last_period = period_specific_state_objects[n_periods - 1]
    resources_last_period = begin_of_period_resources[
        state_objects_last_period["idx_state_of_state_choice"]
    ]

    marg_util_interpolated, value_interpolated, policy_final = solve_final_period(
        state_choice_mat=state_objects_last_period["state_choices"],
        resources=resources_last_period,
        final_period_solution_partial=final_period_solution_partial,
        params=params,
    )

    # Choose which draw we take for policy and value function as those are note
    # saved with respect to the draws
    middle_of_draws = int(len(income_shock_draws) + 1 / 2)
    jnp.save(
        f"endog_grid_{n_periods - 1}.npy",
        resources_last_period[:, :, middle_of_draws],
    )
    jnp.save(f"policy_{n_periods - 1}.npy", policy_final[:, :, middle_of_draws])
    jnp.save(f"value_{n_periods - 1}.npy", value_interpolated[:, :, middle_of_draws])

    for period in range(n_periods - 2, -1, -1):
        state_objects_period = period_specific_state_objects[period]

        # Aggregate the marginal utilities and expected values over all choices and
        # income shock draws
        marg_util, emax = aggregate_marg_utils_exp_values(
            value_state_choice_specific=value_interpolated,
            marg_util_state_choice_specific=marg_util_interpolated,
            reshape_state_choice_vec_to_mat=state_objects_period[
                "reshape_state_choice_vec_to_mat"
            ],
            transform_between_state_and_state_choice_vec=state_objects_period[
                "transform_between_state_and_state_choice_vec"
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
            marg_util=marg_util,
            emax=emax,
            idx_state_choices_period=state_objects_period["idxs_state_choices"],
            map_state_to_post_decision_child_nodes=map_state_to_post_decision_child_nodes,
            exogenous_savings_grid=exog_savings_grid,
            transition_vector_by_state=transition_vector_by_state,
            state_choice_mat=state_objects_period["state_choices"],
            compute_inverse_marginal_utility=compute_inverse_marginal_utility,
            compute_value=compute_value,
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
            state_objects_period["state_choices"][:, -1],
            params,
            compute_value,
        )
        resources_period = begin_of_period_resources[
            state_objects_period["idx_state_of_state_choice"]
        ]

        # ToDo: reorder function arguments
        marg_util_interpolated, value_interpolated = vmap(
            interpolate_and_calc_marginal_utilities,
            in_axes=(None, None, 0, 0, 0, 0, 0, 0, None),
        )(
            compute_marginal_utility,
            compute_value,
            state_objects_period["state_choices"][:, -1],
            resources_period,
            endog_grid_state_choice,
            policy_left_state_choice,
            policy_right_state_choice,
            value_state_choice,
            params,
        )

        jnp.save(f"endog_grid_{period}.npy", endog_grid_state_choice)
        jnp.save(f"policy_left_{period}.npy", policy_left_state_choice)
        jnp.save(f"policy_right_{period}.npy", policy_right_state_choice)
        jnp.save(f"value_{period}.npy", value_state_choice)
