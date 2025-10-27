"""Interface for the DC-EGM algorithm."""

from typing import Any, Callable, Dict, Tuple

import jax
import jax.lax
import jax.numpy as jnp

from dcegm.final_periods import solve_last_two_periods
from dcegm.law_of_motion import calc_cont_grids_next_period
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

    #
    calc_grids_jit = jax.jit(
        lambda income_shock_draws, params_inner: calc_cont_grids_next_period(
            model_structure=model_structure,
            model_config=model_config,
            income_shock_draws_unscaled=income_shock_draws,
            params=params_inner,
            model_funcs=model_funcs,
        )
    )

    cont_grids_next_period = calc_grids_jit(income_shock_draws_unscaled, params)

    (
        value_solved,
        policy_solved,
        endog_grid_solved,
    ) = create_solution_container(
        continuous_states_info=model_config["continuous_states_info"],
        # Read out grid size
        n_total_wealth_grid=model_config["tuning_params"]["n_total_wealth_grid"],
        n_state_choices=model_structure["state_choice_space"].shape[0],
    )

    # Solve the last two periods using lambda to capture static arguments
    solve_last_two_period_jit = jax.jit(
        lambda params_inner, cont_grids, weights, val_solved, pol_solved, endog_solved: solve_last_two_periods(
            params=params_inner,
            continuous_states_info=continuous_states_info,
            cont_grids_next_period=cont_grids,
            income_shock_weights=weights,
            model_funcs=model_funcs,
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
        cont_grids_next_period,
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
        cont_grids_next_period=cont_grids_next_period,
        model_funcs=model_funcs,
        income_shock_weights=income_shock_weights,
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
