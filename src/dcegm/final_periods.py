"""Wrapper to solve the final period of the model."""

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import vmap

from dcegm.check_func_outputs import (
    check_budget_equation_and_return_wealth_plus_optional_aux,
)
from dcegm.law_of_motion import (
    calc_law_of_motion,
    compute_own_continuous_grid_combos,
)
from dcegm.solve_single_period import solve_for_interpolated_values


def solve_last_two_periods(
    params: Dict[str, float],
    continuous_states_info: Dict[str, Any],
    model_structure: Dict[str, Any],
    income_shocks_scaled: jnp.ndarray,
    income_shock_weights: jnp.ndarray,
    model_funcs: Dict[str, Any],
    upper_envelope_method: str,
    skip_endog_grid_storage: bool,
    last_two_period_batch_info: Dict[str, Any],
    value_solved: jnp.ndarray,
    policy_solved: jnp.ndarray,
    endog_grid_solved: Optional[jnp.ndarray],
    debug_info: Optional[Dict[str, bool]],
) -> Tuple[jnp.ndarray, ...]:
    """Solve the final period and the second-to-last period.

    Called once, outside the ``jax.lax.scan`` over the remaining periods (see
    ``backward_induction.py``), because both periods are special:

    1. The final period is solved analytically via ``solve_final_period`` --
       everything is consumed, so there is no continuation value to
       interpolate or Euler equation to invert.
    2. The second-to-last period reaches its own solution the normal EGM way
       (``solve_for_interpolated_values``), but its children (the final
       period) are the ``value``/``marginal_utility`` just computed in step 1
       directly, rather than the *stored*, then re-interpolated, solution
       ``interpolate_value_and_marg_util`` reads for every other period.

    Args:
        params: Model parameters.
        continuous_states_info: ``model_config["continuous_states_info"]``.
        model_structure: Model structure, in particular
            ``state_space_dict``/``state_choice_space_dict``, used to gather
            representative-parent and unique-child-state dicts for the law of
            motion (see ``solve_final_period``).
        income_shocks_scaled: Quadrature points for the income shock, already
            scaled by its mean and standard deviation.
        income_shock_weights: Quadrature weights matching
            ``income_shocks_scaled``.
        model_funcs: Processed model functions.
        upper_envelope_method: ``"fues"`` or ``"druedahl_jorgensen"``.
        skip_endog_grid_storage: Whether the endogenous grid is stored at all;
            see ``endog_grid_solved``.
        last_two_period_batch_info: Batch information for the final and
            second-to-last period -- e.g. which final-period state-choices map
            to which second-to-last-period parents; see
            ``pre_processing/batches/last_two_periods.py``.
        value_solved: The solution container, indexed by state-choice, filled
            so far (empty at this point, since this runs before the main
            scan). Shape ``(n_state_choices, n_continuous_state_combinations,
            n_total_wealth_grid)``.
        policy_solved: Same shape as ``value_solved``.
        endog_grid_solved: Same shape as ``value_solved``, or ``None`` when
            ``skip_endog_grid_storage`` is True.
        debug_info: ``None`` in the normal solve. When given (with
            ``"return_candidates"``), the pre-upper-envelope candidate
            solutions for the second-to-last period are also returned.

    Returns:
        ``(value_solved, policy_solved, endog_grid_solved)`` with the final
        and second-to-last period filled in, plus the three candidate arrays
        when ``debug_info["return_candidates"]`` is True.

    """
    batch_info = last_two_period_batch_info

    (
        value_solved,
        policy_solved,
        endog_grid_solved,
        value_interp_final_period,
        marginal_utility_final_last_period,
    ) = solve_final_period(
        batch_info=batch_info,
        model_structure=model_structure,
        income_shocks_scaled=income_shocks_scaled,
        continuous_states_info=continuous_states_info,
        upper_envelope_method=upper_envelope_method,
        skip_endog_grid_storage=skip_endog_grid_storage,
        params=params,
        model_funcs=model_funcs,
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
    )

    # Check if we have a scalar taste shock scale or state specific. Extract in each of the cases.
    ts_function = model_funcs["taste_shock_function"]
    if ts_function["taste_shock_scale_is_scalar"]:
        taste_shock_scale = ts_function["read_out_taste_shock_scale"](params)
    else:
        taste_shock_scale_per_state_func = ts_function["taste_shock_scale_per_state"]
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
        taste_shock_scale_is_scalar=ts_function["taste_shock_scale_is_scalar"],
        params=params,
        income_shock_weights=income_shock_weights,
        continuous_grids_info=continuous_states_info,
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
    if not skip_endog_grid_storage:
        endog_grid_solved = endog_grid_solved.at[idx_second_last, ...].set(
            out_dict_second_last["endog_grid"]
        )

    # If we do not call the function in debug mode. Return arrays
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
    batch_info: Dict[str, Any],
    model_structure: Dict[str, Any],
    income_shocks_scaled: jnp.ndarray,
    continuous_states_info: Dict[str, Any],
    upper_envelope_method: str,
    skip_endog_grid_storage: bool,
    params: Dict[str, float],
    model_funcs: Dict[str, Any],
    value_solved: jnp.ndarray,
    policy_solved: jnp.ndarray,
    endog_grid_solved: Optional[jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the final period's solution analytically (consumption = wealth).

    The final period's own state-choices act as *children* of the second-to-
    last period's saving choices for the law of motion: ``calc_law_of_motion``
    is called exactly as it would be for any other period, with the final
    period's own state-choices as ``child_state_choices`` and the
    second-to-last period's representative parent supplying the grid the
    transition is evaluated over (see ``rep_parent_idx_per_state``/
    ``rep_parent_idx_per_state_choice`` below). Since nothing is saved in the
    final period, the resulting beginning-of-period wealth is both the
    "child" wealth from that transition and this period's own wealth grid to
    solve over -- there is no further recursion.

    Args:
        batch_info: Final-period batch information -- which state-choices are
            solved, their representative second-to-last-period parents (at
            both state-choice and unique-state granularity), and the
            state-level dedup gather index; see
            ``pre_processing/batches/last_two_periods.py``.
        model_structure: Model structure, in particular
            ``state_space_dict``/``state_choice_space_dict``, read via
            ``batch_info``'s index arrays to build the law-of-motion inputs.
        income_shocks_scaled: Quadrature points for the income shock, already
            scaled by its mean and standard deviation.
        continuous_states_info: ``model_config["continuous_states_info"]``.
        upper_envelope_method: ``"fues"`` or ``"druedahl_jorgensen"``; selects
            which wealth grid (``assets_begin_of_period`` vs.
            ``assets_end_of_period``) the final period's solution is stored
            on.
        skip_endog_grid_storage: Whether the endogenous grid is stored at all;
            see ``endog_grid_solved``.
        params: Model parameters.
        model_funcs: Processed model functions; the final-period-specific
            ``compute_utility_final``/``compute_marginal_utility_final`` are
            used here instead of the regular-period ones.
        value_solved: The solution container, indexed by state-choice, filled
            so far. Shape ``(n_state_choices, n_continuous_state_combinations,
            n_total_wealth_grid)``.
        policy_solved: Same shape as ``value_solved``.
        endog_grid_solved: Same shape as ``value_solved``, or ``None`` when
            ``skip_endog_grid_storage`` is True.

    Returns:
        tuple:

        - value_solved: ``value_solved`` with the final period's rows filled
          in.
        - policy_solved: Likewise for policy.
        - endog_grid_solved: Likewise for the endogenous grid (unchanged when
          ``skip_endog_grid_storage``).
        - value: The final period's own value, shape
          ``(n_final_state_choices, n_continuous_combinations, n_exog_savings,
          n_income_shocks)`` -- fed directly into
          ``solve_for_interpolated_values`` for the second-to-last period,
          bypassing the usual store-then-reinterpolate path.
        - marg_util: The final period's own marginal utility, same shape.

    """

    compute_utility = model_funcs["compute_utility_final"]
    compute_marginal_utility = model_funcs["compute_marginal_utility_final"]

    idx_state_choices_final_period = batch_info["idx_state_choices_final_period"]
    state_choice_mat_final_period = batch_info["state_choice_mat_final_period"]

    has_additional_continuous_states = continuous_states_info[
        "has_additional_continuous_state"
    ]
    additional_continuous_state_names = continuous_states_info[
        "additional_continuous_state_names"
    ]

    # Then call the law of motion to get the continuous states and wealth at the
    # final period. law_of_motion_arrays was assembled once at model setup, keeping
    # only the branch this model takes (see bundle_law_of_motion_arrays in
    # batch_creation.py); the final period's own state-choices are the children.
    final_period_cont_states = calc_law_of_motion(
        law_of_motion_arrays=batch_info["law_of_motion_arrays"],
        state_choice_space_dict=model_structure["state_choice_space_dict"],
        income_shocks_scaled=income_shocks_scaled,
        params=params,
        model_funcs=model_funcs,
        has_additional_continuous_states=has_additional_continuous_states,
        additional_continuous_state_names=additional_continuous_state_names,
    )
    wealth_final_period = final_period_cont_states["assets_begin_of_period"]
    continuous_state_final = final_period_cont_states["continuous_states"]

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
        wealth_final_period,
        params,
        compute_utility,
        compute_marginal_utility,
    )

    if (
        continuous_states_info["has_additional_continuous_state"]
        or upper_envelope_method == "druedahl_jorgensen"
    ):
        # For Druedahl-Jorgensen the storage grid is assets_begin_of_period; for
        # FUES with an additional continuous state it's assets_end_of_period.
        # calc_value_and_budget_for_state_choice handles both, and also the
        # no-additional-continuous-state Druedahl-Jorgensen case (a size-1 dummy
        # combo axis, same convention as solve_euler_equation.py/
        # solve_single_period.py) -- so one branch now covers what used to be two.
        assets_begin = "assets_begin_of_period" in continuous_states_info.keys()

        values_regular, wealth_at_regular = vmap(
            calc_value_and_budget_for_state_choice,
            in_axes=(0, None, None, None, None, None, None),
        )(
            state_choice_mat_final_period,
            model_funcs["continuous_grid_functions"],
            continuous_states_info["additional_continuous_state_names"],
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

        wealth_to_save = wealth_final_period[:, :, :, middle_of_draws]
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

    # Width of the actual final-period wealth grid just built above -- for
    # druedahl_jorgensen this is len(assets_begin_of_period) + 1 (matching
    # n_total_wealth_grid, see check_model_config.py); for fues it's
    # len(assets_grid_end_of_period) + 1.
    n_wealth_final = values_with_zeros.shape[-1]

    value_solved = value_solved.at[
        idx_state_choices_final_period, :, :n_wealth_final
    ].set(values_with_zeros)
    policy_solved = policy_solved.at[
        idx_state_choices_final_period, :, :n_wealth_final
    ].set(wealth_with_zeros)
    if not skip_endog_grid_storage:
        endog_grid_solved = endog_grid_solved.at[
            idx_state_choices_final_period, :, :n_wealth_final
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


def calc_value_and_budget_for_state_choice(
    state_choice_vec,
    continuous_grid_functions,
    additional_continuous_state_names,
    params,
    compute_utility,
    compute_assets_begin_of_period,
    assets_begin,
):
    """Compute the final period's own value/budget for one state-choice.

    Builds this state-choice's own combo axis (what its own solve/storage is indexed
    against) on demand *after* vmapping down to a single state-choice -- the final-
    period analog of solve_euler_equation.py's job for every other period, on its own
    separate code path since the final period has no continuation value and so doesn't
    go through solve_euler_equation.py at all. ``state_choice_vec`` already is each
    row's own identity here (no parent/child ambiguity, we're solving each row's own
    terminal problem), so no representative-parent selection is needed, unlike
    law_of_motion.py's grid selection for a transition *into* a state. Grids live on the
    state-choice space (that's where the solution itself lives), so ``state_choice_vec``
    -- including "choice" -- is exactly the identity a grid may depend on.

    This state-choice's own asset grid (``assets_begin_of_period`` for
    Druedahl-Jorgensen, ``assets_end_of_period`` for FUES) is evaluated on demand
    right here, after vmapping down to a single state-choice, instead of being
    precomputed for the whole batch upfront in a separate vmap and fed in as a
    paired array -- same self-referential pattern as ``own_continuous_state_vec``
    below. The zero point that Druedahl-Jorgensen prepends elsewhere
    (``compute_own_dj_wealth_grid``) is *not* added here: this is the raw
    grid_func output, and the caller appends a single zero point uniformly for
    every branch further down (see ``zeros_to_append`` in ``solve_final_period``).

    With no additional continuous state, ``additional_continuous_state_names`` is
    empty and there is nothing to mesh -- a size-1 ``dummy_cont`` placeholder
    stands in instead, the same convention ``solve_euler_equation.py`` and
    ``solve_single_period.py`` already use, letting this one function (and the
    vmap below) cover the Druedahl-Jorgensen no-additional-continuous-state case
    too, rather than needing a dedicated combo-axis-free sibling. Downstream user
    functions (``compute_utility``) silently ignore the extra ``dummy_cont``
    kwarg, same as they already do on those other paths.

    """
    if additional_continuous_state_names:
        own_continuous_state_vec = compute_own_continuous_grid_combos(
            state_choice_vec,
            continuous_grid_functions,
            additional_continuous_state_names,
        )
    else:
        own_continuous_state_vec = {"dummy_cont": jnp.zeros(1)}
    grid_name = "assets_begin_of_period" if assets_begin else "assets_end_of_period"
    asset_grid = continuous_grid_functions[grid_name](**state_choice_vec)
    return vmap(
        vmap(
            calc_value_and_budget_for_each_gridpoint,
            in_axes=(None, None, 0, None, None, None, None),
        ),
        in_axes=(None, 0, None, None, None, None, None),
    )(
        state_choice_vec,
        own_continuous_state_vec,
        asset_grid,
        params,
        compute_utility,
        compute_assets_begin_of_period,
        assets_begin,
    )


def calc_value_and_budget_for_each_gridpoint(
    state_choice_vec,
    continuous_state_vec,
    asset_grid_point_end_of_previous_period,
    params,
    compute_utility,
    compute_assets_begin_of_period,
    assets_begin,
):
    if assets_begin:
        # If assets begin, the grid is directly the assets we start from
        wealth_final_period = asset_grid_point_end_of_previous_period
    else:
        # "choice" is passed through (not stripped): a budget equation may declare
        # it and return a different budget per choice, same as in law_of_motion.py.
        # Functions that don't declare it are unaffected --
        # determine_function_arguments_and_partial_model_specs filters kwargs down
        # to each function's own signature.
        out_budget = compute_assets_begin_of_period(
            **state_choice_vec,
            **continuous_state_vec,
            asset_end_of_previous_period=asset_grid_point_end_of_previous_period,
            income_shock_previous_period=jnp.array(0.0),
            params=params,
        )
        wealth_final_period = check_budget_equation_and_return_wealth_plus_optional_aux(
            out_budget, optional_aux=False
        )

    value = compute_utility(
        **state_choice_vec,
        **continuous_state_vec,
        wealth=wealth_final_period,
        params=params,
    )

    return value, wealth_final_period
