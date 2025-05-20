from jax import vmap

from dcegm.egm.aggregate_marginal_utility import aggregate_marg_utils_and_exp_values
from dcegm.egm.interpolate_marginal_utility import interpolate_value_and_marg_util
from dcegm.egm.solve_euler_equation import (
    calculate_candidate_solutions_from_euler_equation,
)


def solve_single_period(
    carry,
    xs,
    params,
    continuous_grids_info,
    cont_grids_next_period,
    model_funcs,
    income_shock_weights,
):
    """Solve a single period of the model using DCEGM."""
    (value_solved, policy_solved, endog_grid_solved) = carry

    (
        state_choices_idxs,
        child_state_choices_to_aggr_choice,
        child_states_to_integrate_stochastic,
        child_state_choice_idxs_to_interp,
        child_state_idxs,
        state_choice_mat,
        state_choice_mat_child,
    ) = xs

    # EGM step 1)
    value_interpolated, marginal_utility_interpolated = interpolate_value_and_marg_util(
        model_funcs=model_funcs,
        state_choice_vec=state_choice_mat_child,
        continuous_grids_info=continuous_grids_info,
        cont_grids_next_period=cont_grids_next_period,
        endog_grid_child_state_choice=endog_grid_solved[
            child_state_choice_idxs_to_interp
        ],
        policy_child_state_choice=policy_solved[child_state_choice_idxs_to_interp],
        value_child_state_choice=value_solved[child_state_choice_idxs_to_interp],
        child_state_idxs=child_state_idxs,
        params=params,
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

    endog_grid_state_choice, policy_state_choice, value_state_choice = (
        solve_for_interpolated_values(
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
        )
    )

    value_solved = value_solved.at[state_choices_idxs, :].set(value_state_choice)
    policy_solved = policy_solved.at[state_choices_idxs, :].set(policy_state_choice)
    endog_grid_solved = endog_grid_solved.at[state_choices_idxs, :].set(
        endog_grid_state_choice
    )

    carry = (value_solved, policy_solved, endog_grid_solved)

    return carry, ()


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
):
    # EGM step 2)
    # Aggregate the marginal utilities and expected values over all state-choice
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
        has_second_continuous_state=continuous_grids_info["second_continuous_exists"],
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
        continuous_grid_info=continuous_grids_info,
        state_choice_mat=state_choice_mat,
        compute_utility=model_funcs["compute_utility"],
        params=params,
        discount_factor=discount_factor,
        has_second_continuous_state=continuous_grids_info["second_continuous_exists"],
        compute_upper_envelope_for_state_choice=model_funcs["compute_upper_envelope"],
    )

    return (
        endog_grid_state_choice,
        policy_state_choice,
        value_state_choice,
    )


def run_upper_envelope(
    endog_grid_candidate,
    policy_candidate,
    value_candidate,
    expected_values,
    continuous_grid_info,
    state_choice_mat,
    compute_utility,
    params,
    discount_factor,
    has_second_continuous_state,
    compute_upper_envelope_for_state_choice,
):
    """Run upper envelope to remove suboptimal candidates.

    Vectorized over all state-choice combinations.

    """

    if has_second_continuous_state:
        return vmap(
            vmap(
                compute_upper_envelope_for_state_choice,
                in_axes=(0, 0, 0, 0, 0, None, None, None, None),  # continuous state
            ),
            in_axes=(
                0,
                0,
                0,
                0,
                None,
                0,
                None,
                None,
                None,
            ),  # discrete states and choices
        )(
            endog_grid_candidate,
            policy_candidate,
            value_candidate,
            expected_values[:, :, 0],
            continuous_grid_info["second_continuous_grid"],
            state_choice_mat,
            compute_utility,
            params,
            discount_factor,
        )

    else:
        return vmap(
            compute_upper_envelope_for_state_choice,
            in_axes=(
                0,
                0,
                0,
                0,
                0,
                None,
                None,
                None,
            ),  # discrete states and choice combs
        )(
            endog_grid_candidate,
            policy_candidate,
            value_candidate,
            expected_values[:, 0],
            state_choice_mat,
            compute_utility,
            params,
            discount_factor,
        )
