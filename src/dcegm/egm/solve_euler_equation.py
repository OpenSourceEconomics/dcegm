"""Auxiliary functions for the EGM algorithm."""

from typing import Any, Callable, Dict, Tuple

from jax import numpy as jnp
from jax import vmap


def calculate_candidate_solutions_from_euler_equation(
    continuous_grids_info: Dict[str, Any],
    continuous_state_space: Dict[str, jnp.ndarray],
    marg_util_next: jnp.ndarray,
    emax_next: jnp.ndarray,
    state_choice_mat: Dict[str, jnp.ndarray],
    idx_post_decision_child_states: jnp.ndarray,
    model_funcs: Dict[str, Any],
    params: Dict[str, float],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate candidates for the optimal policy and value function."""

    feasible_marg_utils_child = jnp.take(
        marg_util_next, idx_post_decision_child_states, axis=0
    )
    feasible_emax_child = jnp.take(emax_next, idx_post_decision_child_states, axis=0)

    (
        endog_grid,
        policy,
        value,
        expected_value,
    ) = vmap(
        vmap(
            vmap(
                compute_optimal_policy_and_value,
                in_axes=(1, 1, None, 0, None, None, None),
            ),
            in_axes=(1, 1, 0, None, None, None, None),
        ),
        in_axes=(0, 0, None, None, 0, None, None),
    )(
        feasible_marg_utils_child,
        feasible_emax_child,
        continuous_state_space,
        continuous_grids_info["assets_grid_end_of_period"],
        state_choice_mat,
        model_funcs,
        params,
    )

    return (
        endog_grid,
        value,
        policy,
        expected_value,
    )


def compute_optimal_policy_and_value(
    marg_util_next: jnp.ndarray,
    emax_next: jnp.ndarray,
    continuous_state_vec: Any,
    assets_grid_end_of_period: jnp.ndarray,
    state_choice_vec: Any,
    model_funcs: Dict[str, Any],
    params: Dict[str, float],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute EGM candidates for one state-choice and one continuous-state point.

    Args:
        marg_util_next: Marginal utilities in child states for one assets grid point.
        emax_next: Expected maximum values in child states for one assets grid point.
        continuous_state_vec: Continuous-state values for one continuous-state point.
        assets_grid_end_of_period: Exogenous end-of-period asset grid.
        state_choice_vec: Dictionary of discrete states and choice.
        model_funcs: Processed model functions used by the EGM step.
        params: Model parameter dictionary.

    Returns:
        A tuple ``(endog_grid, policy, value, expected_value)`` where each array is
        state-choice specific on the assets grid.

    """
    state_choice_vec = {**state_choice_vec, **continuous_state_vec}

    compute_inverse_marginal_utility = model_funcs["compute_inverse_marginal_utility"]
    compute_utility = model_funcs["compute_utility"]
    compute_stochastic_transition_vec = model_funcs["compute_stochastic_transition_vec"]

    discount_factor = model_funcs["read_funcs"]["discount_factor"](params)
    interest_rate = model_funcs["read_funcs"]["interest_rate"](params)

    policy, expected_value = solve_euler_equation(
        state_choice_vec=state_choice_vec,
        marg_util_next=marg_util_next,
        emax_next=emax_next,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
        compute_stochastic_transition_vec=compute_stochastic_transition_vec,
        params=params,
        discount_factor=discount_factor,
        interest_rate=interest_rate,
    )
    endog_grid = assets_grid_end_of_period + policy

    utility = compute_utility(consumption=policy, params=params, **state_choice_vec)
    value = utility + discount_factor * expected_value

    return endog_grid, policy, value, expected_value


def solve_euler_equation(
    state_choice_vec: dict,
    marg_util_next: jnp.ndarray,
    emax_next: jnp.ndarray,
    compute_inverse_marginal_utility: Callable,
    compute_stochastic_transition_vec: Callable,
    params: Dict[str, float],
    discount_factor: float,
    interest_rate: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the Euler equation for given discrete choice and child states.

    We integrate over the exogenous process and income uncertainty and
    then apply the inverese marginal utility function.

    Args:
        marg_utils (np.ndarray): 1d array of shape (n_stochastic_states,) containing
            the state-choice specific marginal utilities for a given point on
            the savings grid.
        emax (np.ndarray): 1d array of shape (n_stochastic_states,) containing
            the state-choice specific expected maximum value for a given point on
            the savings grid.
        trans_vec_state (np.ndarray): 1d array of shape (n_stochastic_states,) containing
            for each exogenous process state the corresponding transition probability.
        compute_inverse_marginal_utility (callable): Function for calculating the
            inverse marginal utility, which takes the marginal utility as only input.
            (n_stochastic_states, n_grid_wealth) with the maximum values.
        params (dict): Dictionary of model parameters.

    Returns:
        tuple:

        - policy (np.ndarray): 1d array of the agent's current state- and
            choice-specific consumption policy. Has shape (n_grid_wealth,).
        - expected_value (np.ndarray): 1d array of the agent's current state- and
            choice-specific expected value. Has shape (n_grid_wealth,).

    """

    transition_vec = compute_stochastic_transition_vec(
        params=params, **state_choice_vec
    )

    # Integrate out uncertainty over exogenous processes
    marginal_utility_next = jnp.nansum(transition_vec * marg_util_next)
    expected_value = jnp.nansum(transition_vec * emax_next)

    # RHS of Euler Eq., p. 337 IJRS (2017) by multiplying with marginal wealth
    rhs_euler = marginal_utility_next * (1 + interest_rate) * discount_factor

    policy = compute_inverse_marginal_utility(
        marginal_utility=rhs_euler,
        params=params,
        **state_choice_vec,
    )

    return policy, expected_value
