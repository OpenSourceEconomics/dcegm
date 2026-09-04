"""Interface for the DC-EGM algorithm."""

from typing import Any, Callable, Dict, Tuple

import jax
import jax.lax
import jax.numpy as jnp

from dcegm.final_periods import solve_last_two_periods
from dcegm.pre_processing.sol_container import create_solution_container
from dcegm.solve_single_period import solve_single_period


def backward_induction(
    params: Dict[str, float],
    income_shock_draws_unscaled: jnp.ndarray,
    income_shock_weights: jnp.ndarray,
    model_config: Dict[str, Any],
    model_funcs: Dict[str, Callable],
    model_structure: Dict[str, Any],
    batch_info: Dict[str, Any],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Do backward induction and solve for optimal policy and value function.

    Args:
        params (dict): Dictionary containing the model parameters.
        income_shock_draws_unscaled (np.ndarray): 1d array of shape (n_quad_points,)
            containing the Hermite quadrature points unscaled.
        income_shock_weights (np.ndarrray): 1d array of shape
            (n_stochastic_quad_points) with weights for each stoachstic shock draw.
        model_config (dict): Dictionary containing the model configuration.
        model_funcs (dict): Dictionary containing model functions.
        model_structure (dict): Dictionary containing model structure.
        batch_info (dict): Dictionary containing batch information.

    Returns:
        Tuple: Tuple containing the period-specific endog_grid, policy, and value
            from the backward induction.

    """
    continuous_states_info = model_config["continuous_states_info"]
    skip_endog_grid_storage = model_config["upper_envelope"]["skip_endog_grid_storage"]

    # Scale income shock draws once. This is cheap (shape (n_quad,)) and shared by
    # every batch/period; the actual (large) child continuous-state/wealth
    # transitions are computed on demand per batch/period instead of upfront for the
    # whole state space (see solve_single_period.py / final_periods.py).
    income_shock_mean = model_funcs["read_funcs"]["income_shock_mean"](params)
    income_shock_std = model_funcs["read_funcs"]["income_shock_std"](params)
    income_shocks_scaled = (
        income_shock_draws_unscaled * income_shock_std + income_shock_mean
    )

    n_continuous_state_combinations = continuous_states_info[
        "n_continuous_state_combinations"
    ]

    (
        value_solved,
        policy_solved,
        endog_grid_solved,
    ) = create_solution_container(
        n_continuous_state_combinations=n_continuous_state_combinations,
        # Read out grid size
        n_total_wealth_grid=model_config["n_total_wealth_grid"],
        n_state_choices=model_structure["state_choice_space"].shape[0],
        store_endog_grid=not skip_endog_grid_storage,
    )

    # Solve the last two periods using lambda to capture static arguments
    solve_last_two_period_jit = jax.jit(
        lambda params_inner, shocks_scaled, weights, val_solved, pol_solved, endog_solved: solve_last_two_periods(
            params=params_inner,
            continuous_states_info=continuous_states_info,
            model_structure=model_structure,
            income_shocks_scaled=shocks_scaled,
            income_shock_weights=weights,
            model_funcs=model_funcs,
            upper_envelope_method=model_config["upper_envelope"]["method"],
            skip_endog_grid_storage=skip_endog_grid_storage,
            last_two_period_batch_info=batch_info["last_two_period_info"],
            value_solved=val_solved,
            policy_solved=pol_solved,
            endog_grid_solved=endog_solved,
            debug_info=None,
        )
    )

    (
        value_solved,
        policy_solved,
        endog_grid_solved,
    ) = solve_last_two_period_jit(
        params,
        income_shocks_scaled,
        income_shock_weights,
        value_solved,
        policy_solved,
        endog_grid_solved,
    )

    # If it is a two period model we are done.
    if batch_info["two_period_model"]:
        return value_solved, policy_solved, endog_grid_solved

    # Create JIT-compiled single period solver using lambda
    partial_single_period = lambda carry, xs: solve_single_period(
        carry=carry,
        xs=xs,
        params=params,
        continuous_grids_info=continuous_states_info,
        state_choice_space_dict=model_structure["state_choice_space_dict"],
        state_space_dict=model_structure["state_space_dict"],
        income_shocks_scaled=income_shocks_scaled,
        model_funcs=model_funcs,
        income_shock_weights=income_shock_weights,
        upper_envelope_method=model_config["upper_envelope"]["method"],
        skip_endog_grid_storage=skip_endog_grid_storage,
        debug_info=None,
    )

    for id_segment in range(batch_info["n_segments"]):
        segment_info = batch_info[f"batches_info_segment_{id_segment}"]

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
                segment_info["child_states_to_integrate_stochastic"],
                segment_info["child_state_choice_idxs_to_interp"],
                segment_info["child_states_idxs"],
                segment_info["state_choices"],
                segment_info["state_choices_childs"],
                segment_info["representative_parent_state_choice_idx"],
                segment_info["state_choices_unique_child_states"],
                segment_info["representative_parent_state_choice_idx_per_child_state"],
                segment_info["state_row_for_state_choice"],
            ),
        )

        if not segment_info["batches_cover_all"]:
            last_batch_info = segment_info["last_batch_info"]
            extra_final_carry, () = partial_single_period(
                carry=final_carry,
                xs=(
                    last_batch_info["state_choice_idx"],
                    last_batch_info["child_state_choices_to_aggr_choice"],
                    last_batch_info["child_states_to_integrate_stochastic"],
                    last_batch_info["child_state_choice_idxs_to_interp"],
                    last_batch_info["child_states_idxs"],
                    last_batch_info["state_choices"],
                    last_batch_info["state_choices_childs"],
                    last_batch_info["representative_parent_state_choice_idx"],
                    last_batch_info["state_choices_unique_child_states"],
                    last_batch_info[
                        "representative_parent_state_choice_idx_per_child_state"
                    ],
                    last_batch_info["state_row_for_state_choice"],
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
