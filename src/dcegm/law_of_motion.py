import jax.numpy as jnp
from jax import vmap

from dcegm.check_func_outputs import (
    check_budget_equation_and_return_wealth_plus_optional_aux,
)


def calc_law_of_motion(
    child_state_choices,
    representative_parent_state_choice_vec,
    unique_child_states,
    representative_parent_state_choices_per_child_state,
    state_row_for_state_choice,
    income_shocks_scaled,
    params,
    model_funcs,
    has_additional_continuous_states,
    additional_continuous_state_names,
):
    """Compute the law of motion at whichever granularity is valid for this model.

    Single entry point for every caller (``solve_single_period.py`` via
    ``interpolate_value_and_marg_util``, and ``final_periods.py``), so the
    granularity decision lives here rather than being repeated at each call site.

    The transition into a child does not depend on the child's own *future* choice.
    So unless a user transition function declares ``choice`` -- decided once at
    model-build time by ``_transition_funcs_depend_on_choice`` in
    ``process_model_functions.py`` -- every state-choice sharing a child state would
    compute a bit-identical transition, and it is evaluated once per unique child
    *state* and gathered out instead.

    Both branches run the same transition math: the state-level one is a thin
    dedup/gather wrapper around the state-choice one (see
    ``calc_law_of_motion_for_child_states``). Callers must supply both sets of
    arguments; which one is read depends on the flag.

    """
    if model_funcs["transition_funcs_depend_on_choice"]:
        return calc_law_of_motion_for_state_choices(
            child_state_choices=child_state_choices,
            representative_parent_state_choice_vec=representative_parent_state_choice_vec,
            income_shocks_scaled=income_shocks_scaled,
            params=params,
            model_funcs=model_funcs,
            has_additional_continuous_states=has_additional_continuous_states,
            additional_continuous_state_names=additional_continuous_state_names,
        )

    return calc_law_of_motion_for_child_states(
        child_states=unique_child_states,
        representative_parent_state_choices=(
            representative_parent_state_choices_per_child_state
        ),
        state_row_for_state_choice=state_row_for_state_choice,
        income_shocks_scaled=income_shocks_scaled,
        params=params,
        model_funcs=model_funcs,
        has_additional_continuous_states=has_additional_continuous_states,
        additional_continuous_state_names=additional_continuous_state_names,
    )


def calc_law_of_motion_for_state_choices(
    child_state_choices,
    income_shocks_scaled,
    params,
    model_funcs,
    has_additional_continuous_states,
    additional_continuous_state_names,
    representative_parent_state_choice_vec,
):
    """Compute continuous-state and wealth transitions for a set of state-choices.

    Two *different* state-choice dicts flow in here, and keeping them apart is the
    whole point of this function's signature:

    ``child_state_choices``
        The state-choice whose *beginning-of-period* continuous state/wealth we are
        computing -- i.e. the child, in the main solve path (see
        ``solve_single_period.py``). May or may not carry a ``"choice"`` key; it is
        passed through either way (see the comment below).

    ``representative_parent_state_choice_vec``
        The state-choice whose own continuous grid supplies the *values* fed
        through the law-of-motion function -- in the main solve path, a
        representative parent.

    These are not the same state-choice in general once continuous grids are
    state-choice-specific: the transition function itself correctly depends on the
    child's own identity (e.g. its ``lagged_choice``, which is the parent's
    choice), but the grid values fed into it must come from the parent's own grid.
    This applies to ``assets_end_of_period`` just as much as to the additional
    continuous states: it too is a law-of-motion input (the exogenous grid a
    parent's transition is evaluated over, feeding the child's own budget
    equation), so it is evaluated from the representative parent here too.

    Any one parent works as the representative, because
    ``check_continuous_grid_consistency_across_shared_children`` (run once at
    model-build time) guarantees every parent sharing a child agrees on its own
    grid.

    Callers with no real parent/child relationship to trace -- the whole-state-space
    debug entry point ``calc_cont_grids_next_period`` below -- pass
    ``child_state_choices`` itself here, so each state is its own grid source. That
    is a degenerate but well-defined use of the same contract, and matches the
    global-grid behavior exactly whenever grids are not state-choice-specific.

    """
    # "choice" (the child's own) is passed straight through when present, so a
    # budget equation or continuous-state transition may declare it and get a
    # different transition per choice -- e.g. a choice-specific cost deducted from
    # beginning-of-period wealth. Functions that don't declare it are unaffected:
    # determine_function_arguments_and_partial_model_specs filters kwargs down to
    # each function's own signature. Callers passing a bare state space (no
    # "choice" key at all) are likewise fine, as long as their functions don't ask
    # for it -- which is exactly the condition _transition_funcs_depend_on_choice
    # checks before routing to calc_law_of_motion_for_child_states below.
    state_vec = dict(child_state_choices)

    continuous_state_next_period = _get_continuous_state_next_period(
        has_additional_continuous_states=has_additional_continuous_states,
        child_state_choices=child_state_choices,
        representative_last_period_parent_states=representative_parent_state_choice_vec,
        additional_continuous_state_names=additional_continuous_state_names,
        params=params,
        model_funcs=model_funcs,
    )

    def _transitions_for_one_state(
        states, continuous_state_vec, representative_parent_row
    ):
        # Own assets_end_of_period grid, evaluated here for this one state's
        # representative parent, right next to where it's consumed below --
        # instead of precomputing the whole batch's grids upfront in a separate
        # vmap and feeding the result in as a paired array.
        own_assets_grid_end_of_period = _own_assets_grid_end_of_period_for_one_state(
            representative_parent_row, model_funcs["continuous_grid_functions"]
        )

        def fix_assets_and_shocks_for_broadcast(
            continuous_state_vec,
            asset_end_of_previous_period,
            income_draw,
        ):
            all_states = {**states, **continuous_state_vec}
            return calc_beginning_of_period_assets_for_single_state(
                state_vec=all_states,
                asset_end_of_previous_period=asset_end_of_previous_period,
                income_shock_draw=income_draw,
                params=params,
                compute_assets_begin_of_period=model_funcs[
                    "compute_assets_begin_of_period"
                ],
                aux_outs=False,
            )

        return vmap(
            vmap(
                vmap(
                    fix_assets_and_shocks_for_broadcast,
                    in_axes=(None, None, 0),
                ),
                in_axes=(None, 0, None),
            ),
            in_axes=(0, None, None),
        )(
            continuous_state_vec,
            own_assets_grid_end_of_period,
            income_shocks_scaled,
        )

    assets_begin_of_next_period = vmap(
        _transitions_for_one_state,
        in_axes=(0, 0, 0),
    )(
        state_vec,
        continuous_state_next_period,
        representative_parent_state_choice_vec,
    )

    # Generate result dict
    return {
        "continuous_states": continuous_state_next_period,
        "assets_begin_of_period": assets_begin_of_next_period,
    }


def _own_assets_grid_end_of_period_for_one_state(
    representative_parent_row, continuous_grid_functions
):
    return continuous_grid_functions["assets_end_of_period"](
        **representative_parent_row
    )


def calc_law_of_motion_for_child_states(
    child_states,
    representative_parent_state_choices,
    state_row_for_state_choice,
    income_shocks_scaled,
    params,
    model_funcs,
    has_additional_continuous_states,
    additional_continuous_state_names,
):
    """Law of motion once per unique child *state*, gathered out to state-choices.

    Same computation as ``calc_law_of_motion_for_state_choices`` above -- it *is*
    that function, called with deduplicated child states instead of child
    state-choices, so there is exactly one implementation of the transition math
    for both granularities. Only valid when the user's transition functions do not
    depend on ``choice`` (checked once at model-build time, see
    ``transition_funcs_depend_on_choice`` in ``process_model_functions.py``): that
    function pops ``"choice"`` before calling them anyway, so every state-choice
    sharing a child state would otherwise recompute a bit-identical result
    ``n_choices`` times.

    ``state_row_for_state_choice`` (built in ``child_state_dedup.py``) maps each
    child state-choice back to its row in ``child_states``, so the per-state result
    is expanded to the per-state-choice shape every downstream consumer expects --
    the same gather pattern ``calculate_candidate_solutions_from_euler_equation``
    already uses one stage later.

    """
    law_of_motion_per_state = calc_law_of_motion_for_state_choices(
        child_state_choices=child_states,
        income_shocks_scaled=income_shocks_scaled,
        params=params,
        model_funcs=model_funcs,
        has_additional_continuous_states=has_additional_continuous_states,
        additional_continuous_state_names=additional_continuous_state_names,
        representative_parent_state_choice_vec=representative_parent_state_choices,
    )

    return {
        "continuous_states": {
            name: jnp.take(grid, state_row_for_state_choice, axis=0)
            for name, grid in law_of_motion_per_state["continuous_states"].items()
        },
        "assets_begin_of_period": jnp.take(
            law_of_motion_per_state["assets_begin_of_period"],
            state_row_for_state_choice,
            axis=0,
        ),
    }


def calc_cont_grids_next_period(
    params,
    income_shock_draws_unscaled,
    model_structure,
    model_config,
    model_funcs,
):
    """Compute continuous-state and wealth transitions for the entire state space.

    Thin wrapper around ``calc_law_of_motion_for_state_choices``. Kept only for the
    debug/inspection entry points in ``interfaces/model_class.py`` that need the full
    structure; the main solve path computes this on demand, per batch/period, instead
    (see ``solve_single_period.py``/``final_periods.py``).

    """
    continuous_states_info = model_config["continuous_states_info"]
    state_space_dict = model_structure["state_space_dict"]

    # Scale income shock draws
    income_shock_mean = model_funcs["read_funcs"]["income_shock_mean"](params)
    income_shock_std = model_funcs["read_funcs"]["income_shock_std"](params)
    income_shocks_scaled = (
        income_shock_draws_unscaled * income_shock_std + income_shock_mean
    )

    return calc_law_of_motion_for_state_choices(
        child_state_choices=state_space_dict,
        income_shocks_scaled=income_shocks_scaled,
        params=params,
        model_funcs=model_funcs,
        has_additional_continuous_states=continuous_states_info[
            "has_additional_continuous_state"
        ],
        additional_continuous_state_names=continuous_states_info[
            "additional_continuous_state_names"
        ],
        # No real parent/child relationship to trace here (whole-state-space
        # debug entry point), so use each state's own identity.
        representative_parent_state_choice_vec=state_space_dict,
    )


def _get_continuous_state_next_period(
    has_additional_continuous_states,
    child_state_choices,
    representative_last_period_parent_states,
    additional_continuous_state_names,
    params,
    model_funcs,
):
    if not has_additional_continuous_states:
        # Use an explicit zero-valued dummy continuous state with stable shape
        # (n_states, 1) to keep downstream shapes constant.
        n_states = next(iter(child_state_choices.values())).shape[0]
        dummy_states = {
            "dummy_cont": jnp.zeros((n_states, 1)),
        }
        return dummy_states

    continuous_state_next_period = vmap(
        _continuous_state_next_period_for_one_state,
        in_axes=(0, 0, None, None, None, None),
    )(
        child_state_choices,
        representative_last_period_parent_states,
        model_funcs["continuous_grid_functions"],
        additional_continuous_state_names,
        params,
        model_funcs["next_period_continuous_state"],
    )
    _check_continuous_state_output_keys(
        continuous_state_output=continuous_state_next_period,
        expected_names=additional_continuous_state_names,
    )
    return continuous_state_next_period


def _continuous_state_next_period_for_one_state(
    state_dict,
    representative_last_period_parent_state,
    continuous_grid_functions,
    additional_continuous_state_names,
    params,
    compute_continuous_state,
):
    """Compute one state's beginning-of-period continuous state, across its own grid.

    Builds the grid on demand *after* vmapping down to a single state, instead of
    precomputing the whole batch's grids upfront in a separate vmap and feeding the
    result in as a paired array. Uses ``representative_parent_state_choice_vec`` (the
    representative parent state-choice), not ``state_dict`` (the child) -- see the
    docstring of ``calc_law_of_motion_for_state_choices`` above for why these differ;
    for state-choices without a state-specific grid, ``continuous_grid_functions[name]``
    ignores its input and returns the same global grid for every row, so this is a no-op
    relative to the old behavior in that case.

    """
    own_continuous_state_vec = compute_own_continuous_grid_combos(
        representative_last_period_parent_state,
        continuous_grid_functions,
        additional_continuous_state_names,
    )
    return vmap(
        calc_continuous_state_for_each_grid_point,
        in_axes=(None, 0, None, None),
    )(
        state_dict,
        own_continuous_state_vec,
        params,
        compute_continuous_state,
    )


def compute_own_continuous_grids_raw(
    state_dict,
    continuous_grid_functions,
    additional_continuous_state_names,
):
    """Evaluate one state-choice's own grid for each additional continuous state.

    Unmeshed: each name's 1d grid on its own, not combined into combo points.
    Companion to ``compute_own_continuous_grid_combos`` below (which meshes these),
    used directly by n-D regular-grid interpolation
    (``interpolation/interpnd_regular.py``), which needs each dimension's own grid
    separately -- it combines them into combo points itself, via strides/corner
    tables, rather than a dense meshgrid. Intended to be called via ``vmap`` over a
    batch of state-choices -- one call here is one state-choice's own grids, not
    the whole state-choice space.

    """
    return {
        name: continuous_grid_functions[name](**state_dict)
        for name in additional_continuous_state_names
    }


def compute_own_continuous_grid_combos(
    state_dict,
    continuous_grid_functions,
    additional_continuous_state_names,
):
    """Evaluate one state's own grid for each additional continuous state and mesh them.

    Mirrors the meshgrid+ravel construction in
    ``pre_processing/model_structure/model_structure.py``, but evaluates each name's
    grid on demand via its (possibly state-specific) entry in
    ``continuous_grid_functions`` instead of reading one precomputed global array.
    Intended to be called via ``vmap`` over a batch of states -- one call here is one
    state's own grid, not the whole state space.

    """
    own_grids_dict = compute_own_continuous_grids_raw(
        state_dict, continuous_grid_functions, additional_continuous_state_names
    )
    own_grids = [own_grids_dict[name] for name in additional_continuous_state_names]
    meshed = jnp.meshgrid(*own_grids, indexing="ij")
    return {
        name: grid.ravel()
        for name, grid in zip(additional_continuous_state_names, meshed)
    }


def compute_own_dj_wealth_grid(state_dict, continuous_grid_functions):
    """Evaluate one state-choice's own Druedahl-Jorgensen wealth grid on demand.

    The "m_grid" (``assets_begin_of_period``) with a zero-wealth point prepended -- the
    fixed grid every state-choice's policy/value is stored/interpolated against under
    Druedahl-Jorgensen (see ``check_model_config.py``'s ``dj_wealth_grid``, which this
    replaces with an on-demand, possibly state-choice-specific evaluation). Intended to
    be called via ``vmap`` over a batch of state-choices -- one call here is one state-
    choice's own grid, not the whole state-choice space.

    """
    return jnp.concatenate(
        (
            jnp.zeros(1),
            continuous_grid_functions["assets_begin_of_period"](**state_dict),
        )
    )


def _check_continuous_state_output_keys(
    continuous_state_output,
    expected_names,
):
    expected_keys = set(expected_names)
    output_keys = set(continuous_state_output.keys())
    if output_keys != expected_keys:
        raise ValueError(
            "next_period_continuous_state output keys must match the additional "
            "continuous state names. "
            f"Expected {sorted(expected_keys)}, got {sorted(output_keys)}."
        )


def calc_beginning_of_period_assets_for_single_state(
    state_vec,
    asset_end_of_previous_period,
    income_shock_draw,
    params,
    compute_assets_begin_of_period,
    aux_outs,
):
    out_budget = compute_assets_begin_of_period(
        **state_vec,
        asset_end_of_previous_period=asset_end_of_previous_period,
        income_shock_previous_period=income_shock_draw,
        params=params,
    )
    checked_out = check_budget_equation_and_return_wealth_plus_optional_aux(
        out_budget, optional_aux=aux_outs
    )
    return checked_out


# =====================================================================================
# Second continuous state
# =====================================================================================


def calc_assets_beginning_of_period_2cont_vec(
    state_vec,
    continuous_state_beginning_of_period,
    asset_end_of_previous_period,
    income_shock_draw,
    params,
    compute_assets_begin_of_period,
    aux_outs,
):
    all_states = {
        **state_vec,
        "continuous_state": continuous_state_beginning_of_period,
    }
    checked_out = calc_beginning_of_period_assets_for_single_state(
        state_vec=all_states,
        asset_end_of_previous_period=asset_end_of_previous_period,
        income_shock_draw=income_shock_draw,
        params=params,
        compute_assets_begin_of_period=compute_assets_begin_of_period,
        aux_outs=aux_outs,
    )
    return checked_out


def calculate_continuous_state(
    discrete_states_beginning_of_period,
    continuous_states_end_of_last_period,
    params,
    compute_continuous_state,
):
    """Apply the law of motion for every state against one shared grid (outer product).

    ``continuous_states_end_of_last_period`` is a single grid (shape ``(n_combos,)``
    per continuous-state name), broadcast against every state in
    ``discrete_states_beginning_of_period`` -- i.e. every state is evaluated against
    every grid point. This is the original, still-current contract, used directly
    by ``calc_cont_grids_next_period`` (the whole-state-space debug entry point) and
    exercised directly by ``tests/test_law_of_motion.py``. The main solve path
    instead pairs each state with its own (possibly state-specific) grid, computed
    on demand inside ``_continuous_state_next_period_for_one_state`` above, rather
    than broadcasting one grid shared across all of them.

    """
    continuous_state_beginning_of_period = vmap(
        vmap(
            calc_continuous_state_for_each_grid_point,
            in_axes=(None, 0, None, None),  # continuous state
        ),
        in_axes=(0, None, None, None),  # discrete states
    )(
        discrete_states_beginning_of_period,
        continuous_states_end_of_last_period,
        params,
        compute_continuous_state,
    )
    return continuous_state_beginning_of_period


def calc_continuous_state_for_each_grid_point(
    state_vec,
    continuous_state_vec,
    params,
    compute_continuous_state,
):
    out = compute_continuous_state(
        **state_vec,
        **continuous_state_vec,
        params=params,
    )
    return out


# =====================================================================================
# Simulation
# =====================================================================================


def calculate_assets_begin_of_period_for_all_agents(
    states_beginning_of_period,
    asset_grid_point_end_of_previous_period,
    income_shocks_of_period,
    params,
    compute_assets_begin_of_period,
):
    """Simulation."""
    assets_begin_of_next_period = vmap(
        calc_beginning_of_period_assets_for_single_state,
        in_axes=(0, 0, 0, None, None, None),
    )(
        states_beginning_of_period,
        asset_grid_point_end_of_previous_period,
        income_shocks_of_period,
        params,
        compute_assets_begin_of_period,
        True,
    )
    return assets_begin_of_next_period


def calculate_second_continuous_state_for_all_agents(
    discrete_states_beginning_of_period,
    continuous_state_beginning_of_period,
    params,
    compute_continuous_state,
):
    """Simulation."""
    continuous_state_beginning_of_next_period = vmap(
        calc_continuous_state_for_each_grid_point,
        in_axes=(0, 0, None, None),
    )(
        discrete_states_beginning_of_period,
        continuous_state_beginning_of_period,
        params,
        compute_continuous_state,
    )
    return continuous_state_beginning_of_next_period
