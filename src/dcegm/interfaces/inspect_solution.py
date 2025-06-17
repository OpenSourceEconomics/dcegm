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

    last_relevant_period = model_config["n_periods"] - n_periods - 1

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
        model_config=model_config,
        n_state_choices=relevant_state_choice_space.shape[0],
    )

    if return_candidates:
        (value_candidates, policy_candidates, endog_grid_candidates) = (
            create_solution_container(
                model_config=model_config,
                n_state_choices=relevant_state_choice_space.shape[0],
            )
        )
