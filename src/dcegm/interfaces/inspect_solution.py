import jax.lax
import jax.numpy as jnp
import numpy as np

from dcegm.final_periods import solve_last_two_periods
from dcegm.law_of_motion import calc_cont_grids_next_period
from dcegm.pre_processing.sol_container import create_solution_container


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

    continuous_states_info = model_config["continuous_states_info"]

    cont_grids_next_period = calc_cont_grids_next_period(
        model_structure=model_structure,
        model_config=model_config,
        income_shock_draws_unscaled=income_shock_draws_unscaled,
        params=params,
        model_funcs=model_funcs,
    )

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

    # Create debug information
    debug_info = {
        "return_candidates": return_candidates,
        "rescale_idx": np.where(relevant_state_choices_mask)[0].min(),
    }
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
        last_two_period_batch_info=batch_info["last_two_period_info"],
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        debug_info=debug_info,
    )
    if return_candidates:
        idx_second_last = batch_info["last_two_period_info"][
            "idx_state_choices_second_last_period"
        ]
        idx_second_last_rescaled = idx_second_last - debug_info["rescale_idx"]
        value_candidates = value_candidates.at[idx_second_last_rescaled, ...].set(
            value_candidates_second_last
        )
        policy_candidates = policy_candidates.at[idx_second_last_rescaled, ...].set(
            policy_candidates_second_last,
        )
        endog_grid_candidates = endog_grid_candidates.at[
            idx_second_last_rescaled, ...
        ].set(endog_grid_candidates_second_last)

    return value_solved, policy_solved, endog_grid_solved
