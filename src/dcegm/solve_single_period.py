from dcegm.egm.aggregate_marginal_utility import aggregate_marg_utils_and_exp_values
from dcegm.egm.interpolate_marginal_utility import (
    interpolate_value_and_calc_marginal_utility,
)
from dcegm.egm.solve_euler_equation import (
    calculate_candidate_solutions_from_euler_equation,
)
from jax import vmap


def solve_single_period(
    carry,
    xs,
    params,
    exog_savings_grid,
    income_shock_weights,
    model_funcs,
    compute_upper_envelope,
    taste_shock_scale,
):
    """This function solves a single period of the model using the discrete continous
    endogenous grid method (DCEGM)."""
    value_interpolated, marginal_utility_interpolated = carry

    (
        state_choice_idxs_batch,
        child_state_choice_idxs,
        child_state_ids_per_batch,
        resources_batch,
        state_choice_mat_badge,
    ) = xs

    (
        endog_grid_state_choice,
        policy_left_state_choice,
        policy_right_state_choice,
        value_state_choice,
    ) = solve_for_interpolated_values(
        value_interpolated=value_interpolated,
        marginal_utility_interpolated=marginal_utility_interpolated,
        state_choice_mat_badge=state_choice_mat_badge,
        child_state_ids_per_batch=child_state_ids_per_batch,
        child_state_choice_idxs=child_state_choice_idxs,
        params=params,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
        exog_savings_grid=exog_savings_grid,
        model_funcs=model_funcs,
        compute_upper_envelope=compute_upper_envelope,
    )
    solved_arrays = (
        endog_grid_state_choice,
        policy_left_state_choice,
        policy_right_state_choice,
        value_state_choice,
    )

    # EGM step 1)
    marg_util_interpolated_badge, value_interpolated_badge = vmap(
        interpolate_value_and_calc_marginal_utility,
        in_axes=(None, None, 0, 0, 0, 0, 0, 0, None),
    )(
        model_funcs["compute_marginal_utility"],
        model_funcs["compute_utility"],
        state_choice_mat_badge,
        resources_batch,
        endog_grid_state_choice,
        policy_left_state_choice,
        policy_right_state_choice,
        value_state_choice,
        params,
    )

    value_interpolated = value_interpolated.at[state_choice_idxs_batch, :, :].set(
        value_interpolated_badge
    )
    marginal_utility_interpolated = marginal_utility_interpolated.at[
        state_choice_idxs_batch, :, :
    ].set(marg_util_interpolated_badgechild_state_idxs_second_last_period)

    carry = (value_interpolated, marginal_utility_interpolated)

    return carry, solved_arrays


def solve_for_interpolated_values(
    value_interpolated,
    marginal_utility_interpolated,
    state_choice_mat,
    child_state_idxs,
    states_to_choices_child_states,
    params,
    taste_shock_scale,
    income_shock_weights,
    exog_savings_grid,
    model_funcs,
):
    # EGM step 2)
    # Aggregate the marginal utilities and expected values over all choices and
    # income shock draws
    marg_util, emax = aggregate_marg_utils_and_exp_values(
        value_state_choice_specific=value_interpolated,
        marg_util_state_choice_specific=marginal_utility_interpolated,
        reshape_state_choice_vec_to_mat=states_to_choices_child_states,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
    )

    # EGM step 3)
    (
        endog_grid_candidate,
        value_candidate,
        policy_candidate,
        expected_values,
    ) = calculate_candidate_solutions_from_euler_equation(
        exogenous_savings_grid=exog_savings_grid,
        marg_util=marg_util,
        emax=emax,
        state_choice_vec=state_choice_mat,
        idx_post_decision_child_states=child_state_idxs,
        compute_inverse_marginal_utility=model_funcs[
            "compute_inverse_marginal_utility"
        ],
        compute_utility=model_funcs["compute_utility"],
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        params=params,
    )

    # Run upper envelope to remove suboptimal candidates
    (
        endog_grid_state_choice,
        policy_left_state_choice,
        policy_right_state_choice,
        value_state_choice,
    ) = vmap(
        model_funcs["compute_upper_envelope"],
        in_axes=(0, 0, 0, 0, 0, None, None),  # vmap over state-choice combs
    )(
        endog_grid_candidate,
        policy_candidate,
        value_candidate,
        expected_values[:, 0],
        state_choice_mat,
        params,
        model_funcs["compute_utility"],
    )

    return (
        endog_grid_state_choice,
        policy_left_state_choice,
        policy_right_state_choice,
        value_state_choice,
    )
