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
    exog_savings_grid,
    income_shock_weights,
    resources_beginning_of_period,
    model_funcs,
    taste_shock_scale,
):
    """Solve a single period of the model using the DCEGM method."""
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
        model_funcs["compute_marginal_utility"],
        model_funcs["compute_utility"],
        state_choice_mat_child,
        resources_beginning_of_period[child_state_idxs, :, :],
        endog_grid_solved[child_state_choice_idxs_to_interpolate, :],
        policy_solved[child_state_choice_idxs_to_interpolate, :],
        value_solved[child_state_choice_idxs_to_interpolate, :],
        params,
    )

    (
        endog_grid_state_choice,
        policy_state_choice,
        value_state_choice,
    ) = solve_for_interpolated_values(
        value_interpolated,
        marginal_utility_interpolated,
        state_choice_mat,
        child_states_to_integrate_exog,
        child_state_choices_to_aggr_choice,
        params,
        taste_shock_scale,
        income_shock_weights,
        exog_savings_grid,
        model_funcs,
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
    state_choice_mat,
    compute_utility,
    params,
    compute_upper_envelope_for_state_choice,
):
    """Run upper envelope to remove suboptimal candidates.

    Vectorized over all state-choice combinations.

    """

    return vmap(
        compute_upper_envelope_for_state_choice,
        in_axes=(0, 0, 0, 0, 0, None, None),  # vmap over state-choice combs
    )(
        endog_grid_candidate,
        policy_candidate,
        value_candidate,
        expected_values[:, 0],
        state_choice_mat,
        compute_utility,
        params,
    )
