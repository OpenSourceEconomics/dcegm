"""Auxiliary functions for the EGM algorithm."""

from typing import Any, Callable, Dict, Tuple

from jax import numpy as jnp
from jax import vmap

from dcegm.law_of_motion import compute_own_continuous_grid_combos


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
    """Calculate candidates for the optimal policy and value function.

    This solves/stores each state-choice in ``state_choice_mat``'s *own* problem
    (as opposed to interpolating someone else's child, where the parent/child
    distinction in law_of_motion.py matters) -- so the combo grid used here is
    just each row's own grid, no representative-parent selection needed. See the
    implementation plan at
    docs/source/development/internals/state_specific_continuous_grids_plan.md.

    """
    feasible_marg_utils_child = jnp.take(
        marg_util_next,
        idx_post_decision_child_states,
        axis=0,
        mode="fill",
        fill_value=jnp.nan,
    )
    feasible_emax_child = jnp.take(
        emax_next,
        idx_post_decision_child_states,
        axis=0,
        mode="fill",
        fill_value=jnp.nan,
    )

    (
        endog_grid,
        policy,
        value,
        expected_value,
    ) = vmap(
        compute_optimal_policy_and_value_for_state_choice,
        in_axes=(0, 0, None, 0, None, None, None, None),
    )(
        feasible_marg_utils_child,
        feasible_emax_child,
        continuous_grids_info["assets_grid_end_of_period"],
        state_choice_mat,
        model_funcs,
        params,
        continuous_state_space,
        continuous_grids_info,
    )

    return (
        endog_grid,
        value,
        policy,
        expected_value,
    )


def compute_optimal_policy_and_value_for_state_choice(
    feasible_marg_utils_child: jnp.ndarray,
    feasible_emax_child: jnp.ndarray,
    assets_grid_end_of_period: jnp.ndarray,
    state_choice_vec: Any,
    model_funcs: Dict[str, Any],
    params: Dict[str, float],
    continuous_state_space: Dict[str, jnp.ndarray],
    continuous_grids_info: Dict[str, Any],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute EGM candidates for one state-choice, across its own combo/wealth grid.

    Builds this state-choice's own combo grid on demand *after* vmapping down to a
    single state-choice, instead of precomputing the whole batch's grids upfront in
    a separate vmap and feeding the result in as a paired array. This state-choice
    is solving/storing its *own* problem (not interpolating someone else's child),
    so its own grid is used directly -- no representative-parent selection needed,
    unlike law_of_motion.py's grid selection for a transition *into* a state.
    Grids live on the state-choice space (that's where the solution itself
    lives), so ``state_choice_vec`` -- including "choice" -- is exactly the
    identity a grid may depend on. See the implementation plan at
    docs/source/development/internals/state_specific_continuous_grids_plan.md.

    """
    if continuous_grids_info["has_additional_continuous_state"]:
        own_continuous_state_vec = compute_own_continuous_grid_combos(
            state_choice_vec,
            model_funcs["continuous_grid_functions"],
            continuous_grids_info["additional_continuous_state_names"],
        )
    else:
        # No additional continuous state: continuous_state_space is the
        # {"dummy_cont": ...} placeholder, identical for every state-choice -- keep
        # using it exactly as before.
        own_continuous_state_vec = continuous_state_space

    return vmap(
        vmap(
            compute_optimal_policy_and_value,
            in_axes=(1, 1, None, 0, None, None, None),
        ),
        in_axes=(1, 1, 0, None, None, None, None),
    )(
        feasible_marg_utils_child,
        feasible_emax_child,
        own_continuous_state_vec,
        assets_grid_end_of_period,
        state_choice_vec,
        model_funcs,
        params,
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
