from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax import vmap

from dcegm.egm.aggregate_marginal_utility import aggregate_marg_utils_and_exp_values
from dcegm.egm.interpolate_marginal_utility import interpolate_value_and_marg_util
from dcegm.egm.solve_euler_equation import (
    calculate_candidate_solutions_from_euler_equation,
)
from dcegm.law_of_motion import compute_own_continuous_grid_combos

# The three solution containers threaded through the backward-induction scan as
# its carry: (value_solved, policy_solved, endog_grid_solved). The last is
# `None` when `skip_endog_grid_storage` is True.
SolutionCarry = Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]


def solve_single_period(
    carry: SolutionCarry,
    xs: Tuple[Any, ...],
    params: Dict[str, float],
    continuous_grids_info: Dict[str, Any],
    state_choice_space_dict: Dict[str, jnp.ndarray],
    income_shocks_scaled: jnp.ndarray,
    model_funcs: Dict[str, Any],
    income_shock_weights: jnp.ndarray,
    upper_envelope_method: str,
    skip_endog_grid_storage: bool,
    debug_info: Optional[Dict[str, bool]],
) -> Union[Tuple[SolutionCarry, Tuple[()]], Dict[str, jnp.ndarray]]:
    """Solve one batch of state-choices -- the body of the backward induction scan.

    This is the ``f`` of the ``jax.lax.scan`` in ``backward_induction.py``. One
    call solves every state-choice in one batch, reading the already-solved
    children out of the carry and writing its own results back into it. Batches are
    ordered so that a state-choice's children are always solved in an earlier
    iteration (see ``pre_processing/batches/`` and the batching guide).

    The three EGM steps run in order: interpolate the children's continuation
    values (``interpolate_value_and_marg_util``), aggregate them over choices and
    income shocks, then invert the Euler equation and refine with the upper
    envelope (both in ``solve_for_interpolated_values``).

    Args:
        carry: The solution containers threaded through the scan, as
            ``(value_solved, policy_solved, endog_grid_solved)``. Each is indexed
            by *state-choice* and has shape ``(n_state_choices,
            n_continuous_state_combinations, n_total_wealth_grid)``. They start out
            filled by ``create_solution_container`` and are progressively filled
            in, starting at the final period and working back to period 0.
            ``endog_grid_solved`` is ``None`` when ``skip_endog_grid_storage`` is
            True, because under Druedahl-Jorgensen every state-choice's
            "endogenous" grid is by construction its own ``assets_begin_of_period``
            grid and is recomputed on demand (``compute_own_dj_wealth_grid``)
            instead of stored.
        xs: The per-batch slice produced by the scan. An 11-tuple; all index arrays
            are into the global state-choice space unless noted:

            0. ``state_choices_idxs`` -- the state-choices this batch solves, i.e.
               where its results are written.
            1. ``child_state_choices_to_aggr_choice`` -- for each unique child
               *state*, the positions of its choices in the deduplicated child
               state-choice axis; used to aggregate over choices. Out-of-bounds
               entries mark choices that state does not have.
            2. ``child_states_to_integrate_stochastic`` -- for each (row of this
               batch, stochastic realisation), the position of the resulting child
               *state*; used to integrate over the stochastic transition.
            3. ``child_state_choice_idxs_to_interp`` -- the deduplicated child
               state-choices whose stored solution is read this iteration.
            4. ``child_state_idxs`` -- for each of those, the state it belongs to
               (its own state, dropping the choice).
            5. ``state_choice_mat`` -- state-choice dict for this batch's own rows.
            6. ``state_choice_mat_child`` -- state-choice dict for the children in
               (3).
            7. ``rep_parent_state_choice_idx_per_child_state_choice`` -- for each
               child in (3), the index of one parent state-choice that transitions
               into it. Used *only* to pick whose continuous grid feeds the law of
               motion; any parent works because
               ``check_continuous_grid_consistency_across_shared_children``
               guarantees they agree (see ``law_of_motion.py``).
            8. ``unique_child_states`` -- the children of (3) deduplicated to bare
               *states*, already gathered into a state dict (see
               ``child_state_dedup.py``).
            9. ``rep_parent_state_choice_idx_per_child_state`` -- as (7), but one
               index per unique child state rather than per child state-choice.
            10. ``state_row_for_state_choice`` -- for each child state-choice in
                (3), its row in (8). The gather that expands a per-state result back
                out to per-state-choice.

            Entries (8)-(10) are read only when the law of motion is evaluated at
            state granularity, i.e. when no transition function declares ``choice``
            (see ``calc_law_of_motion``); (7) is read only otherwise -- but which
            branch runs is decided once inside ``calc_law_of_motion``, not here, so
            both sets are always supplied to keep the scan's ``xs`` structure
            model-independent.
        params: Model parameters.
        continuous_grids_info: ``model_config["continuous_states_info"]``.
        state_choice_space_dict: Full state-choice space; ``calc_law_of_motion``
            gathers the representative-parent index arrays (7) and (9) out of this.
        income_shocks_scaled: Quadrature points for the income shock, already
            scaled by its mean and standard deviation.
        model_funcs: Processed model functions.
        income_shock_weights: Quadrature weights matching ``income_shocks_scaled``.
        upper_envelope_method: ``"fues"`` or ``"druedahl_jorgensen"``.
        skip_endog_grid_storage: Whether the endogenous grid is stored at all; see
            ``carry`` above.
        debug_info: ``None`` in the normal solve. When given, the function returns
            a dict rather than a scan-shaped ``(carry, ())`` pair, optionally
            including the pre-upper-envelope candidates.

    Returns:
        ``(carry, ())`` with the updated solution containers -- the empty second
        element because nothing is stacked per scan step. In debug mode a dict is
        returned instead.

    """
    value_solved, policy_solved, endog_grid_solved = carry

    (
        state_choices_idxs,
        child_state_choices_to_aggr_choice,
        child_states_to_integrate_stochastic,
        child_state_choice_idxs_to_interp,
        child_state_idxs,
        state_choice_mat,
        state_choice_mat_child,
        rep_parent_state_choice_idx_per_child_state_choice,
        unique_child_states,
        rep_parent_state_choice_idx_per_child_state,
        state_row_for_state_choice,
    ) = xs

    value_child_state_choice = value_solved[child_state_choice_idxs_to_interp]
    policy_child_state_choice = policy_solved[child_state_choice_idxs_to_interp]
    endog_grid_child_state_choice = (
        None
        if skip_endog_grid_storage
        else endog_grid_solved[child_state_choice_idxs_to_interp]
    )

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
        unique_child_states=unique_child_states,
        rep_parent_state_choice_idx_per_child_state=rep_parent_state_choice_idx_per_child_state,
        rep_parent_state_choice_idx_per_child_state_choice=rep_parent_state_choice_idx_per_child_state_choice,
        state_choice_space_dict=state_choice_space_dict,
        state_row_for_state_choice=state_row_for_state_choice,
    )

    # Check if we have a scalar taste shock scale or state specific. Extract in each of the cases.
    ts_function = model_funcs["taste_shock_function"]
    taste_shock_scale_is_scalar = ts_function["taste_shock_scale_is_scalar"]
    if taste_shock_scale_is_scalar:
        taste_shock_scale = ts_function["read_out_taste_shock_scale"](params)
    else:
        taste_shock_scale_per_state_func = ts_function["taste_shock_scale_per_state"]
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
    value_interpolated: jnp.ndarray,
    marginal_utility_interpolated: jnp.ndarray,
    state_choice_mat: Dict[str, jnp.ndarray],
    child_state_idxs: jnp.ndarray,
    states_to_choices_child_states: jnp.ndarray,
    params: Dict[str, float],
    taste_shock_scale: Union[float, jnp.ndarray],
    taste_shock_scale_is_scalar: bool,
    income_shock_weights: jnp.ndarray,
    continuous_grids_info: Dict[str, Any],
    model_funcs: Dict[str, Any],
    debug_info: Optional[Dict[str, bool]],
) -> Dict[str, jnp.ndarray]:
    """EGM steps 2 and 3: aggregate continuation values, then invert the Euler eq.

    Split out from ``solve_single_period`` because the last two periods reach it by
    a different route: ``final_periods.py`` computes the final period analytically
    (consumption equals wealth) and then calls this directly for the second-to-last
    period, rather than going through the scan.

    Takes the *interpolated* child continuation values -- one entry per child
    state-choice, income shock and continuous-state combination -- and returns the
    current period's solution:

    1. ``aggregate_marg_utils_and_exp_values`` collapses the child-choice axis with
       logit choice probabilities and the income-shock axis with quadrature
       weights, giving one marginal utility and one expected value per child state.
    2. ``calculate_candidate_solutions_from_euler_equation`` gathers those to this
       period's state-choices and inverts the Euler equation, producing candidate
       (endogenous grid, policy, value) triples.
    3. ``run_upper_envelope`` discards the candidates that are not on the upper
       envelope of the value correspondence -- the step that makes this DC-EGM
       rather than plain EGM.

    Args:
        value_interpolated: Child values, shape
            ``(n_child_state_choices, n_continuous_combinations, n_wealth,
            n_income_shocks)``.
        marginal_utility_interpolated: Child marginal utilities, same shape.
        state_choice_mat: State-choice dict for the rows being solved.
        child_state_idxs: For each (row, stochastic realisation), the position of
            the child state -- the stochastic integration map.
        states_to_choices_child_states: For each child state, the positions of its
            choices -- the choice aggregation map.
        params: Model parameters.
        taste_shock_scale: Scalar, or one value per state-choice.
        taste_shock_scale_is_scalar: Which of the two it is.
        income_shock_weights: Quadrature weights for the income shock.
        continuous_grids_info: ``model_config["continuous_states_info"]``.
        model_funcs: Processed model functions.
        debug_info: When given with ``return_candidates``, the pre-upper-envelope
            candidates are added to the returned dict.

    Returns:
        dict with ``"endog_grid"``, ``"policy"`` and ``"value"`` for the rows in
        ``state_choice_mat``, plus the candidate arrays in debug mode.

    """
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
    endog_grid_candidate: jnp.ndarray,
    policy_candidate: jnp.ndarray,
    value_candidate: jnp.ndarray,
    expected_values: jnp.ndarray,
    state_choice_mat: Dict[str, jnp.ndarray],
    compute_utility: Callable,
    params: Dict[str, float],
    discount_factor: float,
    compute_upper_envelope_for_state_choice: Callable,
    continuous_grid_functions: Dict[str, Any],
    continuous_grids_info: Dict[str, Any],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """DC-EGM's refinement step: discard candidates not on the upper envelope.

    The last of the three EGM steps (see ``solve_for_interpolated_values``):
    ``calculate_candidate_solutions_from_euler_equation`` (EGM step 3) can
    produce a non-monotonic, self-intersecting candidate (endogenous grid,
    policy, value) correspondence whenever the discrete choice set makes the
    value function non-concave. This removes the candidates that are not on
    the upper envelope of ``value_candidate`` -- the refinement that makes
    this DC-EGM rather than plain EGM.

    Vectorized over all state-choice combinations. Builds each state-choice's own
    continuous-state combo grid on demand, from its own identity (``state_choice_mat``
    is each row's own state-choice here, no representative-parent selection needed --
    same reasoning as ``solve_euler_equation.py``'s EGM step), instead of reusing one
    grid shared across every state-choice -- a state-choice's own grid may differ once
    continuous grids are state-choice-specific, and the upper-envelope refinement below
    needs to agree with the EGM candidates it is refining.

    Args:
        endog_grid_candidate: Candidate endogenous wealth grid from EGM step 3,
            shape ``(n_state_choices, n_continuous_combinations, n_exog_savings)``.
        policy_candidate: Candidate policy, same shape.
        value_candidate: Candidate value, same shape.
        expected_values: Expected value per state-choice and continuous
            combination, used as the value at zero wealth; only the first
            exogenous-savings entry is read (``[:, :, 0]``) since it is
            constant along that axis.
        state_choice_mat: State-choice dict for the rows being refined -- each
            row's own identity, feeding both its own continuous grid and the
            user's ``compute_utility``/``compute_upper_envelope_for_state_choice``.
        compute_utility: User-supplied utility function.
        params: Model parameters.
        discount_factor: This period's discount factor.
        compute_upper_envelope_for_state_choice: The model's compiled upper-
            envelope routine (FUES or Druedahl-Jorgensen), applied per
            continuous-state combination.
        continuous_grid_functions: Functions returning each (possibly state-
            choice-specific) continuous grid, used to build each row's own
            combo grid on demand.
        continuous_grids_info: ``model_config["continuous_states_info"]``.

    Returns:
        tuple ``(endog_grid, policy, value)``, each shape ``(n_state_choices,
        n_continuous_combinations, n_total_wealth_grid)`` -- the refined
        solution for every state-choice, on the fixed-width storage grid
        (``model_config["upper_envelope"]["tuning_params"]["n_total_wealth_grid"]``)
        the underlying FUES/Druedahl-Jorgensen routine pads or truncates to,
        which need not equal ``n_exog_savings``.

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
