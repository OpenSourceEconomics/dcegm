from jax import vmap

from dcegm.egm.aggregate_marginal_utility import aggregate_marg_utils_and_exp_values
from dcegm.egm.interpolate_marginal_utility import interpolate_value_and_marg_util
from dcegm.egm.solve_euler_equation import (
    calculate_candidate_solutions_from_euler_equation,
)


def solve_single_period(
    carry,
    xs,
    has_second_continuous_state,
    params,
    exog_grids,
    income_shock_weights,
    wealth_and_continuous_state_next_period,
    model_funcs,
    taste_shock_scale,
):
    """Solve a single period of the model using DCEGM."""
    (value_solved, policy_solved, endog_grid_solved) = carry

    (
        state_choices_idxs,
        child_state_choices_to_aggr_choice,
        child_states_to_integrate_exog,
        child_state_choice_idxs_to_interpolate,
        child_state_idxs,
        state_choice_mat,
        state_choice_mat_child,
    ) = xs

    # EGM step 1)
    value_interpolated, marginal_utility_interpolated = interpolate_value_and_marg_util(
        compute_marginal_utility=model_funcs["compute_marginal_utility"],
        compute_utility=model_funcs["compute_utility"],
        state_choice_vec=state_choice_mat_child,
        exog_grids=exog_grids,
        wealth_and_continuous_state_next=wealth_and_continuous_state_next_period,
        endog_grid_child_state_choice=endog_grid_solved[
            child_state_choice_idxs_to_interpolate
        ],
        policy_child_state_choice=policy_solved[child_state_choice_idxs_to_interpolate],
        value_child_state_choice=value_solved[child_state_choice_idxs_to_interpolate],
        child_state_idxs=child_state_idxs,
        has_second_continuous_state=has_second_continuous_state,
        params=params,
    )

    (
        endog_grid_state_choice,
        policy_state_choice,
        value_state_choice,
        marg_util,
        emax,
    ) = solve_for_interpolated_values(
        value_interpolated=value_interpolated,
        marginal_utility_interpolated=marginal_utility_interpolated,
        state_choice_mat=state_choice_mat,
        child_state_idxs=child_states_to_integrate_exog,
        states_to_choices_child_states=child_state_choices_to_aggr_choice,
        params=params,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
        exog_savings_grid=exog_grids["wealth"],
        model_funcs=model_funcs,
        has_second_continuous_state=has_second_continuous_state,
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
    income_shock_weights,
    exog_savings_grid,
    model_funcs,
    has_second_continuous_state,
):
    # EGM step 2)
    # Aggregate the marginal utilities and expected values over all state-choice
    # combinations and income shock draws
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
        exog_savings_grid=exog_savings_grid,
        marg_util_next=marg_util,
        emax_next=emax,
        state_choice_mat=state_choice_mat,
        idx_post_decision_child_states=child_state_idxs,
        compute_inverse_marginal_utility=model_funcs[
            "compute_inverse_marginal_utility"
        ],
        compute_utility=model_funcs["compute_utility"],
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        has_second_continuous_state=has_second_continuous_state,
        params=params,
    )

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
        has_second_continuous_state=has_second_continuous_state,
        compute_upper_envelope_for_state_choice=model_funcs["compute_upper_envelope"],
    )

    return (
        endog_grid_state_choice,
        policy_state_choice,
        value_state_choice,
        marg_util,
        emax,
    )


def run_upper_envelope(
    endog_grid_candidate,
    policy_candidate,
    value_candidate,
    expected_values,
    state_choice_mat,
    compute_utility,
    params,
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
                in_axes=(0, 0, 0, 0, None, None, None),  # continuous state
            ),
            in_axes=(0, 0, 0, 0, 0, None, None),  # discrete states and choices
        )(
            endog_grid_candidate,
            policy_candidate,
            value_candidate,
            expected_values[:, :, 0],
            state_choice_mat,
            compute_utility,
            params,
        )

    else:
        return vmap(
            compute_upper_envelope_for_state_choice,
            in_axes=(0, 0, 0, 0, 0, None, None),  # discrete states and choice combs
        )(
            endog_grid_candidate,
            policy_candidate,
            value_candidate,
            expected_values[:, 0],
            state_choice_mat,
            compute_utility,
            params,
        )
