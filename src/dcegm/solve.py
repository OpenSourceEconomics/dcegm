"""Interface for the DC-EGM algorithm."""

from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax.lax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit

from dcegm.final_periods import solve_last_two_periods
from dcegm.law_of_motion import calc_cont_grids_next_period
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.check_params import process_params
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve_single_period import solve_single_period


def solve_dcegm(
    params: pd.DataFrame,
    options: Dict,
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable] = None,
) -> Dict[int, np.ndarray]:
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
            (i) get the state specific feasible choice set
            (ii) update the endogenous part of the state by the choice
        final_period_solution (callable): User-supplied function for solving the agent's
            last period.
        transition_function (callable): User-supplied function returning for each
            state a transition matrix vector.

    Returns:
        dict: Dictionary containing the period-specific endog_grid, policy, and value
            from the backward induction.

    """

    backward_jit = get_solve_function(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        budget_constraint=budget_constraint,
        utility_functions_final_period=utility_functions_final_period,
    )

    results = backward_jit(params=params)

    return results


def get_solve_function(
    options: Dict[str, Any],
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    utility_functions_final_period: Dict[str, Callable],
    state_space_functions: Dict[str, Callable] = None,
) -> Callable:
    """Create a solve function, which only takes params as input.

    Args:
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

    return get_solve_func_for_model(model=model)


def get_solve_func_for_model(model):
    """Create a solve function, which only takes params as input."""

    options = model["options"]

    exog_grids = options["exog_grids"]
    has_second_continuous_state = len(exog_grids) == 2

    # ToDo: Make interface with several draw possibilities.
    # ToDo: Some day make user supplied draw function.
    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        options["model_params"]["quadrature_points_stochastic"]
    )

    backward_jit = jax.jit(
        partial(
            backward_induction,
            options=options,
            exog_grids=exog_grids,
            has_second_continuous_state=has_second_continuous_state,
            state_space_dict=model["model_structure"]["state_space_dict"],
            n_state_choices=model["model_structure"]["state_choice_space"].shape[0],
            batch_info=model["batch_info"],
            income_shock_draws_unscaled=income_shock_draws_unscaled,
            income_shock_weights=income_shock_weights,
            model_funcs=model["model_funcs"],
        )
    )

    def solve_func(params):
        params_initial = process_params(params)
        return backward_jit(params=params_initial)

    return solve_func


def backward_induction(
    params: Dict[str, float],
    options: Dict[str, Any],
    has_second_continuous_state: bool,
    exog_grids: Dict[str, jnp.ndarray],
    state_space_dict: np.ndarray,
    n_state_choices: int,
    batch_info: Dict[str, np.ndarray],
    income_shock_draws_unscaled: jnp.ndarray,
    income_shock_weights: jnp.ndarray,
    model_funcs: Dict[str, Callable],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Do backward induction and solve for optimal policy and value function.

    Args:
        params (dict): Dictionary containing the model parameters.
        options (dict): Dictionary containing the model options.
        period_specific_state_objects (np.ndarray): Dictionary containing
            period-specific state and state-choice objects, with the following keys:
            - "state_choice_mat" (jnp.ndarray)
            - "idx_state_of_state_choice" (jnp.ndarray)
            - "reshape_state_choice_vec_to_mat" (callable)
            - "transform_between_state_and_state_choice_vec" (callable)
        exog_savings_grid (np.ndarray): 1d array of shape (n_grid_wealth,)
            containing the exogenous savings grid.
        has_second_continuous_state (bool): Boolean indicating whether the model
            features a second continuous state variable. If False, the only
            continuous state variable is consumption/savings.
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
        dict: Dictionary containing the period-specific endog_grid, policy, and value
            from the backward induction.

    """
    taste_shock_scale = params["lambda"]

    cont_grids_next_period = calc_cont_grids_next_period(
        state_space_dict=state_space_dict,
        exog_grids=exog_grids,
        income_shock_draws_unscaled=income_shock_draws_unscaled,
        params=params,
        model_funcs=model_funcs,
        has_second_continuous_state=has_second_continuous_state,
    )

    # Create solution containers. The 20 percent extra in wealth grid needs to go
    # into tuning parameters
    (
        value_solved,
        policy_solved,
        endog_grid_solved,
    ) = create_solution_container(
        n_state_choices=n_state_choices,
        options=options,
        has_second_continuous_state=has_second_continuous_state,
    )

    # Solve the last two periods. We do this separately as the marginal utility of
    # the child states in the last period is calculated from the marginal utility
    # function of the bequest function, which might differ.
    (
        value_solved,
        policy_solved,
        endog_grid_solved,
    ) = solve_last_two_periods(
        cont_grids_next_period=cont_grids_next_period,
        params=params,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
        exog_grids=exog_grids,
        model_funcs=model_funcs,
        last_two_period_batch_info=batch_info["last_two_period_info"],
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        has_second_continuous_state=has_second_continuous_state,
    )

    # If it is a two period model we are done.
    if batch_info["two_period_model"]:
        return value_solved, policy_solved, endog_grid_solved

    def partial_single_period(carry, xs):
        return solve_single_period(
            carry=carry,
            xs=xs,
            has_second_continuous_state=has_second_continuous_state,
            params=params,
            exog_grids=exog_grids,
            cont_grids_next_period=cont_grids_next_period,
            income_shock_weights=income_shock_weights,
            model_funcs=model_funcs,
            taste_shock_scale=taste_shock_scale,
        )

    segment_info = batch_info["batches_info_segment_0"]

    carry_start = (
        value_solved,
        policy_solved,
        endog_grid_solved,
    )

    final_carry, _ = jax.lax.scan(
        f=partial_single_period,
        init=carry_start,
        xs=(
            segment_info["batches_state_choice_idx"],
            segment_info["child_state_choices_to_aggr_choice"],
            segment_info["child_states_to_integrate_exog"],
            segment_info["child_state_choice_idxs_to_interp"],
            segment_info["child_states_idxs"],
            segment_info["state_choices"],
            segment_info["state_choices_childs"],
        ),
    )

    if not segment_info["batches_cover_all"]:
        last_batch_info = segment_info["last_batch_info"]
        extra_final_carry, () = partial_single_period(
            carry=final_carry,
            xs=(
                last_batch_info["state_choice_idx"],
                last_batch_info["child_state_choices_to_aggr_choice"],
                last_batch_info["child_states_to_integrate_exog"],
                last_batch_info["child_state_choice_idxs_to_interp"],
                last_batch_info["child_states_idxs"],
                last_batch_info["state_choices"],
                last_batch_info["state_choices_childs"],
            ),
        )

        (
            value_solved,
            policy_solved,
            endog_grid_solved,
        ) = extra_final_carry
    else:
        (
            value_solved,
            policy_solved,
            endog_grid_solved,
        ) = final_carry

    return (
        value_solved,
        policy_solved,
        endog_grid_solved,
    )


def create_solution_container(n_state_choices, options, has_second_continuous_state):
    """Create solution containers for value, policy, and endog_grid."""

    n_total_wealth_grid = options["tuning_params"]["n_total_wealth_grid"]

    if has_second_continuous_state:
        n_second_continuous_grid = options["tuning_params"]["n_second_continuous_grid"]

        value_solved = jnp.full(
            (n_state_choices, n_second_continuous_grid, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
        policy_solved = jnp.full(
            (n_state_choices, n_second_continuous_grid, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
        endog_grid_solved = jnp.full(
            (n_state_choices, n_second_continuous_grid, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
    else:
        value_solved = jnp.full(
            (n_state_choices, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
        policy_solved = jnp.full(
            (n_state_choices, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )
        endog_grid_solved = jnp.full(
            (n_state_choices, n_total_wealth_grid),
            dtype=jnp.float64,
            fill_value=jnp.nan,
        )

    return value_solved, policy_solved, endog_grid_solved
