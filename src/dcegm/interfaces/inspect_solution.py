import copy

import jax.lax
import jax.numpy as jnp
import numpy as np

from dcegm.final_periods import solve_last_two_periods
from dcegm.law_of_motion import calc_cont_grids_next_period
from dcegm.pre_processing.sol_container import create_solution_container
from dcegm.solve_single_period import solve_single_period


def partially_solve(
    income_shock_draws_unscaled,
    income_shock_weights,
    model_config,
    batch_info,
    model_funcs,
    model_structure,
    params,
    n_periods,
    return_candidates=False,
):
    """Partially solve the model for the last n_periods.

    This method allows for large models to only solve part of the model, to debug the solution process.

    Args:
        params: Model parameters.
        n_periods: Number of periods to solve.
        return_candidates: If True, additionally return candidate solutions before applying the upper envelope.

    """
    batch_info_internal = copy.deepcopy(batch_info)

    if n_periods < 2:
        raise ValueError("You must at least solve for two periods.")

    continuous_states_info = model_config["continuous_states_info"]

    cont_grids_next_period = calc_cont_grids_next_period(
        model_structure=model_structure,
        model_config=model_config,
        income_shock_draws_unscaled=income_shock_draws_unscaled,
        params=params,
        model_funcs=model_funcs,
    )
    # Determine the last period we need to solve for.
    last_relevant_period = model_config["n_periods"] - n_periods

    relevant_state_choices_mask = (
        model_structure["state_choice_space"][:, 0] >= last_relevant_period
    )
    relevant_state_choice_space = model_structure["state_choice_space"][
        relevant_state_choices_mask
    ]

    (
        value_solved,
        policy_solved,
        endog_grid_solved,
    ) = create_solution_container(
        continuous_states_info=model_config["continuous_states_info"],
        # Read out grid size
        n_total_wealth_grid=model_config["tuning_params"]["n_total_wealth_grid"],
        n_state_choices=relevant_state_choice_space.shape[0],
    )

    if return_candidates:
        n_assets_end_of_period = model_config["continuous_states_info"][
            "assets_grid_end_of_period"
        ].shape[0]
        (value_candidates, policy_candidates, endog_grid_candidates) = (
            create_solution_container(
                continuous_states_info=model_config["continuous_states_info"],
                n_total_wealth_grid=n_assets_end_of_period,
                n_state_choices=relevant_state_choice_space.shape[0],
            )
        )

    # Determine rescale idx for reduced solution
    rescale_idx = np.where(relevant_state_choices_mask)[0].min()

    # Create debug information
    debug_info = {
        "return_candidates": return_candidates,
    }
    last_two_period_batch_info = batch_info_internal["last_two_period_info"]
    # Rescale the indexes to save of the last two periods:
    last_two_period_batch_info["idx_state_choices_final_period"] = (
        last_two_period_batch_info["idx_state_choices_final_period"] - rescale_idx
    )
    last_two_period_batch_info["idx_state_choices_second_last_period"] = (
        last_two_period_batch_info["idx_state_choices_second_last_period"] - rescale_idx
    )
    (
        value_solved,
        policy_solved,
        endog_grid_solved,
        value_candidates_second_last,
        policy_candidates_second_last,
        endog_grid_candidates_second_last,
    ) = solve_last_two_periods(
        params=params,
        continuous_states_info=continuous_states_info,
        cont_grids_next_period=cont_grids_next_period,
        income_shock_weights=income_shock_weights,
        model_funcs=model_funcs,
        last_two_period_batch_info=last_two_period_batch_info,
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        debug_info=debug_info,
    )
    if return_candidates:
        idx_second_last = batch_info_internal["last_two_period_info"][
            "idx_state_choices_second_last_period"
        ]
        value_candidates = value_candidates.at[idx_second_last, ...].set(
            value_candidates_second_last
        )
        policy_candidates = policy_candidates.at[idx_second_last, ...].set(
            policy_candidates_second_last,
        )
        endog_grid_candidates = endog_grid_candidates.at[idx_second_last, ...].set(
            endog_grid_candidates_second_last
        )

    if n_periods <= 2:
        out_dict = {
            "value": value_solved,
            "policy": policy_solved,
            "endog_grid": endog_grid_solved,
        }
        if return_candidates:
            out_dict["value_candidates"] = value_candidates
            out_dict["policy_candidates"] = policy_candidates
            out_dict["endog_grid_candidates"] = endog_grid_candidates

        return out_dict

    stop_segment_loop = False
    for id_segment in range(batch_info_internal["n_segments"]):
        segment_info = batch_info_internal[f"batches_info_segment_{id_segment}"]

        n_batches_in_segment = segment_info["batches_state_choice_idx"].shape[0]

        for id_batch in range(n_batches_in_segment):
            periods_batch = segment_info["state_choices"]["period"][id_batch, :]

            # Now there can be three cases:
            # 1) All periods are smaller than the last relevant period. Then we stop the loop
            # 2) Part of the periods are smaller than the last relevant period. Then we only solve for the partial state choices.
            # 3) All periods are larger than the last relevant period. Then we solve for state choices.
            if (periods_batch < last_relevant_period).all():
                stop_segment_loop = True
                break
            elif (periods_batch < last_relevant_period).any():
                solve_mask = periods_batch >= last_relevant_period
                state_choices_batch = {
                    key: segment_info["state_choices"][key][id_batch, solve_mask]
                    for key in segment_info["state_choices"].keys()
                }
                # We need to rescale the idx, because of saving
                idx_to_solve = (
                    segment_info["batches_state_choice_idx"][id_batch, solve_mask]
                    - rescale_idx
                )
                child_states_to_integrate_stochastic = segment_info[
                    "child_states_to_integrate_stochastic"
                ][id_batch, solve_mask, :]

            else:
                state_choices_batch = {
                    key: segment_info["state_choices"][key][id_batch, :]
                    for key in segment_info["state_choices"].keys()
                }
                # We need to rescale the idx, because of saving
                idx_to_solve = (
                    segment_info["batches_state_choice_idx"][id_batch, :] - rescale_idx
                )
                child_states_to_integrate_stochastic = segment_info[
                    "child_states_to_integrate_stochastic"
                ][id_batch, :, :]

            state_choices_childs_batch = {
                key: segment_info["state_choices_childs"][key][id_batch, :]
                for key in segment_info["state_choices_childs"].keys()
            }
            xs = (
                idx_to_solve,
                segment_info["child_state_choices_to_aggr_choice"][id_batch, :, :],
                child_states_to_integrate_stochastic,
                segment_info["child_state_choice_idxs_to_interp"][id_batch, :],
                segment_info["child_states_idxs"][id_batch, :],
                state_choices_batch,
                state_choices_childs_batch,
            )
            carry = (value_solved, policy_solved, endog_grid_solved)
            single_period_out_dict = solve_single_period(
                carry=carry,
                xs=xs,
                params=params,
                continuous_grids_info=continuous_states_info,
                cont_grids_next_period=cont_grids_next_period,
                model_funcs=model_funcs,
                income_shock_weights=income_shock_weights,
                debug_info=debug_info,
            )

            value_solved = single_period_out_dict["value"]
            policy_solved = single_period_out_dict["policy"]
            endog_grid_solved = single_period_out_dict["endog_grid"]

            # If candidates are requested, we assign them to the solution container
            if return_candidates:
                value_candidates = value_candidates.at[idx_to_solve, ...].set(
                    single_period_out_dict["value_candidates"]
                )
                policy_candidates = policy_candidates.at[idx_to_solve, ...].set(
                    single_period_out_dict["policy_candidates"]
                )
                endog_grid_candidates = endog_grid_candidates.at[idx_to_solve, ...].set(
                    single_period_out_dict["endog_grid_candidates"]
                )

        if stop_segment_loop:
            break

    out_dict = {
        "value": value_solved,
        "policy": policy_solved,
        "endog_grid": endog_grid_solved,
    }
    if return_candidates:
        out_dict["value_candidates"] = value_candidates
        out_dict["policy_candidates"] = policy_candidates
        out_dict["endog_grid_candidates"] = endog_grid_candidates
    return out_dict
