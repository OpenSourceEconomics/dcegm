"""Wrapper to solve the final period of the model."""

from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import vmap

from dcegm.solve_single_period import solve_for_interpolated_values


def solve_last_two_periods(
    params: Dict[str, float],
    continuous_states_info: Dict[str, Any],
    model_structure: Dict[str, Any],
    cont_grids_next_period: Dict[str, Any],
    income_shock_weights: jnp.ndarray,
    model_funcs: Dict[str, Any],
    upper_envelope_method: str,
    last_two_period_batch_info,
    value_solved,
    policy_solved,
    endog_grid_solved,
    debug_info,
):
    """Solves the last two periods of the model.

    The last two periods are solved using the EGM algorithm. The last period is
    solved using the user-specified utility function and the second to last period
    is solved using the user-specified utility function and the user-specified
    bequest function.

    Args:
        wealth_beginning_of_period (np.ndarray): 2d array of shape
            (n_states, n_grid_wealth) of the wealth at the beginning of the
            period.
        params (dict): Dictionary of model parameters.
        compute_utility (callable): User supplied utility function.
        compute_marginal_utility (callable): User supplied marginal utility
            function.
        last_two_period_batch_info (dict): Dictionary containing information about the batch
            size and the state space.
        value_solved (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the value function for
            all states, end of period assets, and income shocks.
        endog_grid_solved (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the endogenous grid
            for all states, end of period assets, and income shocks.

    """
    batch_info = last_two_period_batch_info
    (
        value_solved,
        policy_solved,
        endog_grid_solved,
        value_interp_final_period,
        marginal_utility_final_last_period,
    ) = solve_final_period(
        idx_state_choices_final_period=batch_info["idx_state_choices_final_period"],
        idx_parent_states_final_period=batch_info["idxs_parent_states_final_period"],
        state_choice_mat_final_period=batch_info["state_choice_mat_final_period"],
        cont_grids_next_period=cont_grids_next_period,
        continuous_states_info=continuous_states_info,
        upper_envelope_method=upper_envelope_method,
        model_structure=model_structure,
        params=params,
        model_funcs=model_funcs,
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
    )

    # Check if we have a scalar taste shock scale or state specific. Extract in each of the cases.
    if model_funcs["taste_shock_function"]["taste_shock_scale_is_scalar"]:
        taste_shock_scale = model_funcs["taste_shock_function"][
            "read_out_taste_shock_scale"
        ](params)
    else:
        taste_shock_scale_per_state_func = model_funcs["taste_shock_function"][
            "taste_shock_scale_per_state"
        ]
        taste_shock_scale = vmap(taste_shock_scale_per_state_func, in_axes=(0, None))(
            last_two_period_batch_info["state_choice_mat_final_period"], params
        )

    out_dict_second_last = solve_for_interpolated_values(
        value_interpolated=value_interp_final_period,
        marginal_utility_interpolated=marginal_utility_final_last_period,
        state_choice_mat=last_two_period_batch_info[
            "state_choice_mat_second_last_period"
        ],
        child_state_idxs=last_two_period_batch_info["child_states_second_last_period"],
        states_to_choices_child_states=last_two_period_batch_info[
            "state_to_choices_final_period"
        ],
        taste_shock_scale=taste_shock_scale,
        taste_shock_scale_is_scalar=model_funcs["taste_shock_function"][
            "taste_shock_scale_is_scalar"
        ],
        params=params,
        income_shock_weights=income_shock_weights,
        continuous_grids_info=continuous_states_info,
        continuous_state_space=model_structure["continuous_state_space"],
        model_funcs=model_funcs,
        debug_info=debug_info,
    )

    idx_second_last = last_two_period_batch_info["idx_state_choices_second_last_period"]

    value_solved = value_solved.at[idx_second_last, ...].set(
        out_dict_second_last["value"]
    )
    policy_solved = policy_solved.at[idx_second_last, ...].set(
        out_dict_second_last["policy"]
    )
    endog_grid_solved = endog_grid_solved.at[idx_second_last, ...].set(
        out_dict_second_last["endog_grid"]
    )

    # If we do not call the function in debug mode. Assign everything and return
    if debug_info is None:
        return (
            value_solved,
            policy_solved,
            endog_grid_solved,
        )

    else:
        # If candidates are also needed to returned we return them additionally to the solution containers.
        if debug_info["return_candidates"]:
            return (
                value_solved,
                policy_solved,
                endog_grid_solved,
                out_dict_second_last["value_candidates"],
                out_dict_second_last["policy_candidates"],
                out_dict_second_last["endog_grid_candidates"],
            )

        else:
            return (
                value_solved,
                policy_solved,
                endog_grid_solved,
            )


def solve_final_period(
    idx_state_choices_final_period,
    idx_parent_states_final_period,
    state_choice_mat_final_period,
    cont_grids_next_period: Dict[str, Any],
    continuous_states_info: Dict[str, Any],
    upper_envelope_method: str,
    model_structure: Dict[str, Any],
    params: Dict[str, float],
    model_funcs: Dict[str, Any],
    value_solved,
    policy_solved,
    endog_grid_solved,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute solution to final period for policy and value function.

    In the last period, everything is consumed, i.e. consumption = savings.
    Args:


    Returns:
        tuple:
        - marginal_utilities_choices (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the marginal utility of
            consumption for all final states, end of period assets, and
            income shocks.
        - final_value (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the optimal
            value function for all final states, end of period assets, and
            income shocks.
        - final_policy (np.ndarray): 3d array of shape
            (n_states, n_grid_wealth, n_income_shocks) of the optimal
            policy for all final states, end of period assets, and
            income shocks.

    """

    compute_utility = model_funcs["compute_utility_final"]
    compute_marginal_utility = model_funcs["compute_marginal_utility_final"]

    wealth_child_states_final_period = cont_grids_next_period["assets_begin_of_period"][
        idx_parent_states_final_period
    ]
    continuous_state_final = {
        key: value[idx_parent_states_final_period]
        for key, value in cont_grids_next_period["continuous_states"].items()
    }

    n_assets = wealth_child_states_final_period.shape[-2]

    value, marg_util = vmap(
        vmap(
            vmap(
                vmap(
                    calc_value_and_marg_util_for_each_gridpoint,
                    in_axes=(None, None, 0, None, None, None),
                ),
                in_axes=(None, None, 0, None, None, None),
            ),
            in_axes=(None, 0, 0, None, None, None),
        ),
        in_axes=(0, 0, 0, None, None, None),
    )(
        state_choice_mat_final_period,
        continuous_state_final,
        wealth_child_states_final_period,
        params,
        compute_utility,
        compute_marginal_utility,
    )

    if continuous_states_info["has_additional_continuous_state"]:
        # We also need to solve at the state space and the child states to store correctly
        # For Druedahl Jorgensen wealth needs to be assets_begin_of_period
        assets_begin = continuous_states_info["assets_begin_of_period"]
        if upper_envelope_method == "druedahl_jorgensen":
            asset_grid = continuous_states_info["assets_begin_of_period"]
        else:
            asset_grid = continuous_states_info["assets_grid_end_of_period"]

        values_regular, wealth_at_regular = vmap(
            vmap(
                vmap(
                    calc_value_and_budget_for_each_gridpoint,
                    in_axes=(None, None, 0, None, None, None),
                ),
                in_axes=(None, 0, None, None, None, None),
            ),
            in_axes=(0, None, None, None, None, None),
        )(
            state_choice_mat_final_period,
            model_structure["continuous_state_space"],
            asset_grid,
            params,
            compute_utility,
            model_funcs["compute_assets_begin_of_period"],
            assets_begin,
        )

        sort_idx = jnp.argsort(wealth_at_regular, axis=2)
        wealth_sorted = jnp.take_along_axis(wealth_at_regular, sort_idx, axis=2)
        values_sorted = jnp.take_along_axis(values_regular, sort_idx, axis=2)
    else:
        middle_of_draws = int((value.shape[3] - 1) / 2)
        value_final = value[:, :, :, middle_of_draws]

        wealth_to_save = wealth_child_states_final_period[:, :, :, middle_of_draws]
        sort_idx = jnp.argsort(wealth_to_save, axis=2)
        wealth_sorted = jnp.take_along_axis(wealth_to_save, sort_idx, axis=2)
        values_sorted = jnp.take_along_axis(value_final, sort_idx, axis=2)

    zeros_to_append = jnp.zeros(values_sorted.shape[:-1])

    values_with_zeros = jnp.concatenate(
        (zeros_to_append[..., None], values_sorted), axis=2
    )
    wealth_with_zeros = jnp.concatenate(
        (zeros_to_append[..., None], wealth_sorted), axis=2
    )

    value_solved = value_solved.at[
        idx_state_choices_final_period, :, : n_assets + 1
    ].set(values_with_zeros)
    policy_solved = policy_solved.at[
        idx_state_choices_final_period, :, : n_assets + 1
    ].set(wealth_with_zeros)
    endog_grid_solved = endog_grid_solved.at[
        idx_state_choices_final_period, :, : n_assets + 1
    ].set(wealth_with_zeros)

    return (
        value_solved,
        policy_solved,
        endog_grid_solved,
        value,
        marg_util,
    )


def calc_value_and_marg_util_for_each_gridpoint(
    state_choice_vec,
    continuous_state_vec,
    wealth,
    params,
    compute_utility,
    compute_marginal_utility,
):
    all_states = {**state_choice_vec, **continuous_state_vec}
    value = compute_utility(
        **all_states,
        wealth=wealth,
        params=params,
    )

    marg_util = compute_marginal_utility(
        **all_states,
        wealth=wealth,
        params=params,
    )

    return value, marg_util


def calc_value_and_budget_for_each_gridpoint(
    state_choice_vec,
    continuous_state_vec,
    asset_grid_point_end_of_previous_period,
    params,
    compute_utility,
    compute_assets_begin_of_period,
    assets_begin,
):
    state_vec = state_choice_vec.copy()
    state_vec.pop("choice")

    if assets_begin:
        # If assets begin, the grid is directly the assets we start from
        wealth_final_period = asset_grid_point_end_of_previous_period
    else:
        wealth_final_period = compute_assets_begin_of_period(
            **state_vec,
            **continuous_state_vec,
            asset_end_of_previous_period=asset_grid_point_end_of_previous_period,
            income_shock_previous_period=jnp.array(0.0),
            params=params,
        )

    value = compute_utility(
        **state_choice_vec,
        **continuous_state_vec,
        wealth=wealth_final_period,
        params=params,
    )

    return value, wealth_final_period
