import jax.numpy as jnp
from jax import vmap

from dcegm.egm.aggregate_marginal_utility import aggregate_marg_utils_and_exp_values
from dcegm.egm.interpolate_marginal_utility import interpolate_value_and_marg_util
from dcegm.egm.solve_euler_equation import (
    calculate_candidate_solutions_from_euler_equation,
)
from dcegm.law_of_motion import compute_own_continuous_grid_combos


def solve_single_period(
    carry,
    xs,
    params,
    continuous_grids_info,
    state_choice_space_dict,
    state_space_dict,
    income_shocks_scaled,
    model_funcs,
    income_shock_weights,
    upper_envelope_method,
    skip_endog_grid_storage,
    debug_info,
):
    """Solve a single period of the model using DCEGM."""
    value_solved, policy_solved, endog_grid_solved = carry

    (
        state_choices_idxs,
        child_state_choices_to_aggr_choice,
        child_states_to_integrate_stochastic,
        child_state_choice_idxs_to_interp,
        child_state_idxs,
        state_choice_mat,
        state_choice_mat_child,
        representative_parent_state_choice_idx,
        unique_child_states,
        representative_parent_state_choice_idx_per_child_state,
        state_row_for_state_choice,
    ) = xs

    value_child_state_choice = value_solved[child_state_choice_idxs_to_interp]
    policy_child_state_choice = policy_solved[child_state_choice_idxs_to_interp]
    endog_grid_child_state_choice = (
        None
        if skip_endog_grid_storage
        else endog_grid_solved[child_state_choice_idxs_to_interp]
    )

    # A representative parent's own state-choice, for each of this batch's
    # deduplicated children -- used only to pick which state-choice's own
    # continuous grid feeds the law of motion (see law_of_motion.py). Grids live on
    # the state-choice space (that's where the solution itself lives), so this is a
    # state-choice index, not a bare state. Any one parent works:
    # check_continuous_grid_consistency_across_shared_children (run once at
    # model-build time) guarantees every parent sharing a child agrees on its own
    # grid.
    representative_parent_state_choice_dict = {
        key: var[representative_parent_state_choice_idx]
        for key, var in state_choice_space_dict.items()
    }

    # Same, at the coarser unique-child-*state* granularity, for the law-of-motion
    # fast path taken when the user's transition functions don't depend on "choice"
    # (see interpolate_value_and_marg_util / law_of_motion.py).
    representative_parent_state_choice_dict_per_child_state = {
        key: var[representative_parent_state_choice_idx_per_child_state]
        for key, var in state_choice_space_dict.items()
    }

    # EGM step 1)
    value_interpolated, marginal_utility_interpolated = interpolate_value_and_marg_util(
        model_funcs=model_funcs,
        child_state_choices=state_choice_mat_child,
        continuous_grids_info=continuous_grids_info,
        income_shocks_scaled=income_shocks_scaled,
        endog_grid_child_state_choice=endog_grid_child_state_choice,
        policy_child_state_choice=policy_child_state_choice,
        value_child_state_choice=value_child_state_choice,
        params=params,
        upper_envelope_method=upper_envelope_method,
        skip_endog_grid_storage=skip_endog_grid_storage,
        representative_parent_state_choice_vec=representative_parent_state_choice_dict,
        unique_child_states=unique_child_states,
        representative_parent_state_choices_per_child_state=(
            representative_parent_state_choice_dict_per_child_state
        ),
        state_row_for_state_choice=state_row_for_state_choice,
    )

    # Check if we have a scalar taste shock scale or state specific. Extract in each of the cases.
    taste_shock_scale_is_scalar = model_funcs["taste_shock_function"][
        "taste_shock_scale_is_scalar"
    ]
    if taste_shock_scale_is_scalar:
        taste_shock_scale = model_funcs["taste_shock_function"][
            "read_out_taste_shock_scale"
        ](params)
    else:
        taste_shock_scale_per_state_func = model_funcs["taste_shock_function"][
            "taste_shock_scale_per_state"
        ]
        taste_shock_scale = vmap(taste_shock_scale_per_state_func, in_axes=(0, None))(
            state_choice_mat_child, params
        )

    out_dict_period = solve_for_interpolated_values(
        value_interpolated=value_interpolated,
        marginal_utility_interpolated=marginal_utility_interpolated,
        state_choice_mat=state_choice_mat,
        child_state_idxs=child_states_to_integrate_stochastic,
        states_to_choices_child_states=child_state_choices_to_aggr_choice,
        params=params,
        taste_shock_scale=taste_shock_scale,
        taste_shock_scale_is_scalar=taste_shock_scale_is_scalar,
        income_shock_weights=income_shock_weights,
        continuous_grids_info=continuous_grids_info,
        model_funcs=model_funcs,
        debug_info=debug_info,
    )
    value_solved = value_solved.at[state_choices_idxs, :].set(out_dict_period["value"])
    policy_solved = policy_solved.at[state_choices_idxs, :].set(
        out_dict_period["policy"]
    )
    if not skip_endog_grid_storage:
        endog_grid_solved = endog_grid_solved.at[state_choices_idxs, :].set(
            out_dict_period["endog_grid"]
        )

    # If we are not in the debug mode, we only return the solution as a tuple and an empty tuple.
    if debug_info is None:
        carry = (value_solved, policy_solved, endog_grid_solved)
        return carry, ()

    else:
        # In debug mode we return a dictionary.
        out_dict = {
            "value": value_solved,
            "policy": policy_solved,
            "endog_grid": endog_grid_solved,
        }

        # If candidates are requested, we add them
        if debug_info["return_candidates"]:
            out_dict = {
                **out_dict,
                "value_candidates": out_dict_period["value_candidates"],
                "policy_candidates": out_dict_period["policy_candidates"],
                "endog_grid_candidates": out_dict_period["endog_grid_candidates"],
            }
        return out_dict


def solve_for_interpolated_values(
    value_interpolated,
    marginal_utility_interpolated,
    state_choice_mat,
    child_state_idxs,
    states_to_choices_child_states,
    params,
    taste_shock_scale,
    taste_shock_scale_is_scalar,
    income_shock_weights,
    continuous_grids_info,
    model_funcs,
    debug_info,
):

    # EGM step 2)
    # Aggregate the marginal utilities and expected values over all child state-choice
    # combinations and income shock draws
    marg_util, emax = aggregate_marg_utils_and_exp_values(
        value_state_choice_specific=value_interpolated,
        marg_util_state_choice_specific=marginal_utility_interpolated,
        reshape_state_choice_vec_to_mat=states_to_choices_child_states,
        taste_shock_scale=taste_shock_scale,
        taste_shock_scale_is_scalar=taste_shock_scale_is_scalar,
        income_shock_weights=income_shock_weights,
    )

    # EGM step 3)
    (
        endog_grid_candidate,
        value_candidate,
        policy_candidate,
        expected_values,
    ) = calculate_candidate_solutions_from_euler_equation(
        continuous_grids_info=continuous_grids_info,
        marg_util_next=marg_util,
        emax_next=emax,
        state_choice_mat=state_choice_mat,
        idx_post_decision_child_states=child_state_idxs,
        model_funcs=model_funcs,
        params=params,
    )

    discount_factor = model_funcs["read_funcs"]["discount_factor"](params)

    # Run upper envelope over all state-choice combinations to remove suboptimal
    # candidates
    (
        endog_grid_state_choice,
        policy_state_choice,
        value_state_choice,
    ) = run_upper_envelope(
        endog_grid_candidate=endog_grid_candidate,
        policy_candidate=policy_candidate,
        value_candidate=value_candidate,
        expected_values=expected_values,
        state_choice_mat=state_choice_mat,
        compute_utility=model_funcs["compute_utility"],
        params=params,
        discount_factor=discount_factor,
        compute_upper_envelope_for_state_choice=model_funcs["compute_upper_envelope"],
        continuous_grid_functions=model_funcs["continuous_grid_functions"],
        continuous_grids_info=continuous_grids_info,
    )
    out_dict = {
        "endog_grid": endog_grid_state_choice,
        "policy": policy_state_choice,
        "value": value_state_choice,
    }

    # If candidates are requested, we additionally return them in the output dictionary.
    if debug_info is not None:
        if debug_info["return_candidates"]:
            out_dict["endog_grid_candidates"] = endog_grid_candidate
            out_dict["policy_candidates"] = policy_candidate
            out_dict["value_candidates"] = value_candidate

    return out_dict


def run_upper_envelope(
    endog_grid_candidate,
    policy_candidate,
    value_candidate,
    expected_values,
    state_choice_mat,
    compute_utility,
    params,
    discount_factor,
    compute_upper_envelope_for_state_choice,
    continuous_grid_functions,
    continuous_grids_info,
):
    """Run upper envelope to remove suboptimal candidates.

    Vectorized over all state-choice combinations. Builds each state-choice's own
    continuous-state combo grid on demand, from its own identity (``state_choice_mat``
    is each row's own state-choice here, no representative-parent selection needed --
    same reasoning as ``solve_euler_equation.py``'s EGM step), instead of reusing one
    grid shared across every state-choice -- a state-choice's own grid may differ once
    continuous grids are state-choice-specific, and the upper-envelope refinement below
    needs to agree with the EGM candidates it is refining.

    """
    return vmap(
        _run_upper_envelope_for_state_choice,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None),
    )(
        endog_grid_candidate,
        policy_candidate,
        value_candidate,
        expected_values[:, :, 0],
        state_choice_mat,
        compute_utility,
        params,
        discount_factor,
        compute_upper_envelope_for_state_choice,
        continuous_grid_functions,
        continuous_grids_info["additional_continuous_state_names"],
        continuous_grids_info["has_additional_continuous_state"],
    )


def _run_upper_envelope_for_state_choice(
    endog_grid_candidate_state_choice,
    policy_candidate_state_choice,
    value_candidate_state_choice,
    expected_values_state_choice,
    state_choice_vec,
    compute_utility,
    params,
    discount_factor,
    compute_upper_envelope_for_state_choice,
    continuous_grid_functions,
    additional_continuous_state_names,
    has_additional_continuous_state,
):
    if has_additional_continuous_state:
        own_continuous_state_vec = compute_own_continuous_grid_combos(
            state_choice_vec,
            continuous_grid_functions,
            additional_continuous_state_names,
        )
    else:
        own_continuous_state_vec = {"dummy_cont": jnp.zeros(1)}

    return vmap(
        compute_upper_envelope_for_state_choice,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None),
    )(
        endog_grid_candidate_state_choice,
        policy_candidate_state_choice,
        value_candidate_state_choice,
        expected_values_state_choice,
        own_continuous_state_vec,
        state_choice_vec,
        compute_utility,
        params,
        discount_factor,
    )
