import jax.numpy as jnp
from jax import vmap

from dcegm.check_func_outputs import (
    check_budget_equation_and_return_wealth_plus_optional_aux,
)


def calc_law_of_motion_for_state_choices(
    state_choice_vec,
    income_shocks_scaled,
    params,
    model_funcs,
    has_additional_continuous_states,
    additional_continuous_state_names,
    grid_source_state_choice_vec,
):
    """Compute continuous-state and wealth transitions for a set of state-choices.

    ``state_choice_vec`` may or may not contain a ``"choice"`` key. It is dropped (via a
    no-op-if-absent pop) before being passed to the user-supplied law-of-motion
    functions, since the transition does not depend on it -- this is what lets
    ``calc_cont_grids_next_period`` below reuse this function unchanged with the full
    (choice-less) state space.

    ``state_choice_vec`` is the state-choice we are computing the *beginning-of-period*
    continuous state/wealth for -- i.e. the child, in the main solve path (see
    ``solve_single_period.py``). ``grid_source_state_choice_vec`` is a *separate*
    state-choice dict: the state-choice whose own continuous grid supplies the
    values fed through the law-of-motion function (in the main solve path, a
    representative parent). These are not the same state-choice in general once
    continuous grids are state-choice-specific: the transition function itself
    correctly depends on the child's own identity (e.g. its ``lagged_choice``,
    which is the parent's choice), but the grid *values* fed into it must come
    from the source state-choice's own grid. Callers with no real parent/child
    relationship to trace (e.g. the whole-state-space debug entry point below)
    pass ``state_choice_vec`` itself here (including "choice" -- grids live on
    the state-choice space, so it's a legitimate part of the identity), matching
    today's global-grid behavior exactly. This applies to ``assets_end_of_period``
    just as much as to the additional continuous states: it too is a law-of-motion
    input (the exogenous grid a parent's transition is evaluated over, feeding the
    child's own budget equation), so it is evaluated from
    ``grid_source_state_choice_vec`` here too, not from ``state_choice_vec``.

    """
    state_vec = dict(state_choice_vec)
    state_vec.pop("choice", None)

    continuous_state_next_period = _get_continuous_state_next_period(
        has_additional_continuous_states=has_additional_continuous_states,
        state_space_dict=state_vec,
        grid_source_state_choice_vec=grid_source_state_choice_vec,
        additional_continuous_state_names=additional_continuous_state_names,
        params=params,
        model_funcs=model_funcs,
    )
    own_assets_grid_end_of_period = vmap(
        _own_assets_grid_end_of_period_for_one_state,
        in_axes=(0, None),
    )(grid_source_state_choice_vec, model_funcs["continuous_grid_functions"])

    def fix_assets_and_shocks_for_broadcast(
        states,
        continuous_state_vec,
        asset_end_of_previous_period,
        income_draw,
    ):
        all_states = {**states, **continuous_state_vec}
        assets_begin_of_period = calc_beginning_of_period_assets_for_single_state(
            state_vec=all_states,
            asset_end_of_previous_period=asset_end_of_previous_period,
            income_shock_draw=income_draw,
            params=params,
            compute_assets_begin_of_period=model_funcs[
                "compute_assets_begin_of_period"
            ],
            aux_outs=False,
        )
        return assets_begin_of_period

    assets_begin_of_next_period = vmap(
        vmap(
            vmap(
                vmap(
                    fix_assets_and_shocks_for_broadcast,
                    in_axes=(None, None, None, 0),
                ),
                in_axes=(None, None, 0, None),
            ),
            in_axes=(None, 0, None, None),
        ),
        in_axes=(0, 0, 0, None),
    )(
        state_vec,
        continuous_state_next_period,
        own_assets_grid_end_of_period,
        income_shocks_scaled,
    )

    # Generate result dict
    return {
        "continuous_states": continuous_state_next_period,
        "assets_begin_of_period": assets_begin_of_next_period,
    }


def _own_assets_grid_end_of_period_for_one_state(
    grid_source_row, continuous_grid_functions
):
    return continuous_grid_functions["assets_end_of_period"](**grid_source_row)


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
        state_choice_vec=state_space_dict,
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
        grid_source_state_choice_vec=state_space_dict,
    )


def _get_continuous_state_next_period(
    has_additional_continuous_states,
    state_space_dict,
    grid_source_state_choice_vec,
    additional_continuous_state_names,
    params,
    model_funcs,
):
    if not has_additional_continuous_states:
        # Use an explicit zero-valued dummy continuous state with stable shape
        # (n_states, 1) to keep downstream shapes constant.
        n_states = next(iter(state_space_dict.values())).shape[0]
        dummy_states = {
            "dummy_cont": jnp.zeros((n_states, 1)),
        }
        return dummy_states

    continuous_state_next_period = vmap(
        _continuous_state_next_period_for_one_state,
        in_axes=(0, 0, None, None, None, None),
    )(
        state_space_dict,
        grid_source_state_choice_vec,
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
    grid_source_state_choice_vec,
    continuous_grid_functions,
    additional_continuous_state_names,
    params,
    compute_continuous_state,
):
    """Compute one state's beginning-of-period continuous state, across its own grid.

    Builds the grid on demand *after* vmapping down to a single state, instead of
    precomputing the whole batch's grids upfront in a separate vmap and feeding the
    result in as a paired array. Uses ``grid_source_state_choice_vec`` (the
    representative parent state-choice), not ``state_dict`` (the child) -- see the
    docstring of ``calc_law_of_motion_for_state_choices`` above for why these differ;
    for state-choices without a state-specific grid, ``continuous_grid_functions[name]``
    ignores its input and returns the same global grid for every row, so this is a no-op
    relative to the old behavior in that case.

    """
    own_continuous_state_vec = compute_own_continuous_grid_combos(
        grid_source_state_choice_vec,
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
