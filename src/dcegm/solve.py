"""Interface for the DC-EGM algorithm."""
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.budget import calculate_resources
from dcegm.egm.aggregate_marginal_utility import (
    aggregate_marg_utils_and_exp_values,
)
from dcegm.egm.interpolate_marginal_utility import (
    interpolate_value_and_calc_marginal_utility,
)
from dcegm.egm.solve_euler_equation import (
    calculate_candidate_solutions_from_euler_equation,
)
from dcegm.final_period import solve_final_period
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.params import process_params
from dcegm.pre_processing.setup_model import setup_model
from jax import jit
from jax import vmap


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict,
    exog_savings_grid: jnp.ndarray,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
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
            (i) get the state specific feasible choice set
            (ii) update the endogenous part of the state by the choice
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
        utility_functions_final_period=utility_functions_final_period,
    )

    results = backward_jit(params=params)

    return results


def get_solve_function(
    options: Dict[str, Any],
    exog_savings_grid: jnp.ndarray,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    utility_functions_final_period: Dict[str, Callable],
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
        utility_functions_final_period (Dict[str, callable]): Dictionary of two
            user-supplied utility functions for the last period:
            (i) utility
            (ii) marginal utility
    Returns:
        callable: The partial solve function that only takes ```params``` as input.

    """
    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    return get_solve_func_for_model(
        model=model,
        exog_savings_grid=exog_savings_grid,
        options=options,
    )


def get_solve_func_for_model(model, exog_savings_grid, options):
    n_periods = options["state_space"]["n_periods"]

    # ToDo: Make interface with several draw possibilities.
    # ToDo: Some day make user supplied draw function.
    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        options["model_params"]["quadrature_points_stochastic"]
    )

    backward_jit = jit(
        partial(
            backward_induction,
            period_specific_state_objects=model["period_specific_state_objects"],
            exog_savings_grid=exog_savings_grid,
            state_space=model["state_space"],
            income_shock_draws_unscaled=income_shock_draws_unscaled,
            income_shock_weights=income_shock_weights,
            n_periods=n_periods,
            model_funcs=model["model_funcs"],
            compute_upper_envelope=model["compute_upper_envelope"],
        )
    )

    def solve_func(params):
        params_initial = process_params(params)
        return backward_jit(params=params_initial)

    return solve_func


def backward_induction(
    params: Dict[str, float],
    period_specific_state_objects: Dict[int, Dict[str, np.ndarray]],
    exog_savings_grid: np.ndarray,
    state_space: np.ndarray,
    income_shock_draws_unscaled: np.ndarray,
    income_shock_weights: np.ndarray,
    n_periods: int,
    model_funcs: Dict[str, Callable],
    compute_upper_envelope: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Do backward induction and solve for optimal policy and value function.

    Args:
        params (dict): Dictionary containing the model parameters.
        period_specific_state_objects (np.ndarray): Dictionary containing
            period-specific state and state-choice objects, with the following keys:
            - "state_choice_mat" (jnp.ndarray)
            - "idx_state_of_state_choice" (jnp.ndarray)
            - "reshape_state_choice_vec_to_mat" (callable)
            - "transform_between_state_and_state_choice_vec" (callable)
        exog_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
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
        income_shock_draws_unscaled (np.ndarray): 1d array of shape (n_quad_points,)
            containing the Hermite quadrature points unscaled.
        income_shock_weights (np.ndarrray): 1d array of shape
            (n_stochastic_quad_points) with weights for each stoachstic shock draw.
        n_periods (int): Number of periods.
        model_funcs (dict): Dictionary containing following model functions:
            - compute_marginal_utility (callable): User-defined function to compute the
                agent's marginal utility. The input ```params``` is already partialled
                in.
            - compute_inverse_marginal_utility (Callable): Function for calculating the
                inverse marginal utiFality, which takes the marginal utility as only
                 input.
            - compute_next_period_wealth (callable): User-defined function to compute
                the agent's wealth of the next period (t + 1). The inputs
                ```saving```, ```shock```, ```params``` and ```options```
                are already partialled in.
            - transition_vector_by_state (Callable): Partialled transition function
                return transition vector for each state.
            - final_period_partial (Callable): Partialled function for calculating the
                consumption as well as value function and marginal utility in the final
                period.
        compute_upper_envelope (Callable): Function for calculating the upper
                envelope of the policy and value function. If the number of discrete
                choices is 1, this function is a dummy function that returns the policy
                and value function as is, without performing a fast upper envelope
                scan.

    Returns:
        dict: Dictionary containing the period-specific endog_grid, policy_left,
            policy_right, and value from the backward induction.

    """

    taste_shock_scale = params["lambda"]

    resources_beginning_of_period = calculate_resources(
        states_beginning_of_period=state_space,
        savings_end_of_last_period=exog_savings_grid,
        income_shocks_of_period=income_shock_draws_unscaled * params["sigma"],
        params=params,
        compute_beginning_of_period_resources=model_funcs[
            "compute_beginning_of_period_resources"
        ],
    )

    (
        value,
        policy_left,
        policy_right,
        endog_grid,
        value_interpolated_next_period,
        marg_util_interpolated_next_period,
    ) = solve_final_period(
        state_objects=period_specific_state_objects[n_periods - 1],
        resources_beginning_of_period=resources_beginning_of_period,
        params=params,
        compute_utility=model_funcs["compute_utility_final"],
        compute_marginal_utility=model_funcs["compute_marginal_utility_final"],
    )

    for period in range(n_periods - 2, -1, -1):
        state_objects_period = period_specific_state_objects[period]
        resources_period = resources_beginning_of_period[
            state_objects_period["idx_parent_states"]
        ]
        reshape_state_choice_vec_to_mat_prev_period = period_specific_state_objects[
            period + 1
        ]["reshape_state_choice_vec_to_mat"]
        (
            endog_grid_period,
            policy_left_period,
            policy_right_period,
            value_period,
            marg_util_interpolated_next_period,
            value_interpolated_next_period,
        ) = solve_single_period(
            value_interpolated_previous_period=value_interpolated_next_period,
            marg_util_interpolated_previous_period=marg_util_interpolated_next_period,
            params=params,
            state_objects=state_objects_period,
            reshape_state_choice_vec_to_mat_prev_period=reshape_state_choice_vec_to_mat_prev_period,
            exog_savings_grid=exog_savings_grid,
            resources_period=resources_period,
            income_shock_weights=income_shock_weights,
            model_funcs=model_funcs,
            compute_upper_envelope=compute_upper_envelope,
            taste_shock_scale=taste_shock_scale,
        )
        value = jnp.append(value_period, value, axis=0)
        policy_left = jnp.append(policy_left_period, policy_left, axis=0)
        policy_right = jnp.append(policy_right_period, policy_right, axis=0)
        endog_grid = jnp.append(endog_grid_period, endog_grid, axis=0)

    return value, policy_left, policy_right, endog_grid


def solve_single_period(
    value_interpolated_previous_period: jnp.ndarray,
    marg_util_interpolated_previous_period: jnp.ndarray,
    params: Dict[str, float],
    state_objects: Dict[str, np.ndarray],
    reshape_state_choice_vec_to_mat_prev_period,
    exog_savings_grid: np.ndarray,
    resources_period: jnp.ndarray,
    income_shock_weights: jnp.ndarray,
    model_funcs: Dict[str, Callable],
    compute_upper_envelope: Callable,
    taste_shock_scale: float,
):
    # EGM step 2)
    # Aggregate the marginal utilities and expected values over all choices and
    # income shock draws
    marg_util, emax = aggregate_marg_utils_and_exp_values(
        value_state_choice_specific=value_interpolated_previous_period,
        marg_util_state_choice_specific=marg_util_interpolated_previous_period,
        reshape_state_choice_vec_to_mat=reshape_state_choice_vec_to_mat_prev_period,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
    )

    # EGM step 3)
    (
        endog_grid_candidate,
        value_candidate,
        policy_candidate,
        expected_values,
    ) = calculate_candidate_solutions_from_euler_equation(
        exogenous_savings_grid=exog_savings_grid,
        marg_util=marg_util,
        emax=emax,
        state_choice_vec=state_objects["state_choice_mat"],
        idx_post_decision_child_states=state_objects["idx_feasible_child_nodes"],
        compute_inverse_marginal_utility=model_funcs[
            "compute_inverse_marginal_utility"
        ],
        compute_utility=model_funcs["compute_utility"],
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
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
        state_objects["state_choice_mat"],
        params,
        model_funcs["compute_utility"],
    )

    # EGM step 1)
    marg_util_interpolated, value_interpolated = vmap(
        interpolate_value_and_calc_marginal_utility,
        in_axes=(None, None, 0, 0, 0, 0, 0, 0, None),
    )(
        model_funcs["compute_marginal_utility"],
        model_funcs["compute_utility"],
        state_objects["state_choice_mat"],
        resources_period,
        endog_grid_state_choice,
        policy_left_state_choice,
        policy_right_state_choice,
        value_state_choice,
        params,
    )

    return (
        endog_grid_state_choice,
        policy_left_state_choice,
        policy_right_state_choice,
        value_state_choice,
        marg_util_interpolated,
        value_interpolated,
    )
