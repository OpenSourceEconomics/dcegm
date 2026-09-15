"""State-specific continuous state grids.

Grids live on the *state-choice* space, not the bare state space: that's where
the solution itself lives (``value_solved``/``policy_solved``/``endog_grid_solved``
are indexed by state-choice, see ``create_solution_container``), so a grid may
depend on the discrete state *and* on ``choice``.

"""

import numpy as np


def evaluate_state_specific_continuous_grids(
    state_choice_space,
    discrete_state_choice_names,
    continuous_grid_functions,
    state_specific_names,
    continuous_states_info,
):
    """Evaluate each user-supplied continuous grid function once per state-choice.

    Only evaluates ``state_specific_names`` -- the continuous states the user
    actually made state-specific via the ``continuous_grid_functions`` argument to
    ``create_model_dict`` (see ``process_continuous_grid_functions``); names left at
    their default (constant) grid are skipped, since they are trivially consistent
    across every state-choice and don't need checking.

    Also validates that every state-choice's grid has the expected length for that
    name. For names with a declared default array in
    ``model_config["continuous_states"]``, that's the declared length. For names
    declared as ``None`` (no default array at all -- fully state-choice-specific),
    there is nothing to read a length from yet, so it is *pinned* here instead: the
    first state-choice's grid_func evaluation fixes the length, and every other
    state-choice's grid is validated against that. Either way, storage containers
    are rectangular (same number of grid points for every state-choice, only the
    values differ), so a mismatched length would otherwise silently corrupt shapes
    downstream instead of failing here, close to the user-supplied function that
    caused it.

    This is a one-time NumPy-side pass over the full state-choice space, done once
    during model-structure construction. The grids are used to build the
    child-sharing consistency check (see
    ``check_continuous_grid_consistency_across_shared_children`` below) and are not
    stored in ``model_structure`` or threaded through solving. The pinned lengths
    (for names declared as ``None``) *are* returned, for the caller to merge back
    into ``continuous_states_info`` (e.g. ``n_continuous_state_combinations``,
    ``n_total_wealth_grid``), since those couldn't be computed any earlier than
    this, the first point a real state-choice exists to evaluate against.

    """
    if len(state_specific_names) == 0:
        return {}, {}

    expected_lengths = _expected_grid_lengths(continuous_states_info)
    resolved_lengths = {}

    grids_per_state_choice = {}
    for name in state_specific_names:
        grid_func = continuous_grid_functions[name]
        pinning_size = name not in expected_lengths

        grid_rows = []
        for row in state_choice_space:
            state_choice_dict = {
                key: row[i] for i, key in enumerate(discrete_state_choice_names)
            }
            grid = np.asarray(grid_func(**state_choice_dict))

            if pinning_size:
                expected_lengths[name] = grid.shape[0] if grid.ndim == 1 else -1
                resolved_lengths[name] = expected_lengths[name]
                pinning_size = False

            expected_length = expected_lengths[name]
            if grid.ndim != 1 or grid.shape[0] != expected_length:
                length_source = (
                    f"the length of the first state-choice evaluated, since "
                    f"model_config['continuous_states']['{name}'] is None"
                    if name in resolved_lengths
                    else "the length declared in "
                    f"model_config['continuous_states']['{name}']"
                )
                raise ValueError(
                    f"\n\nThe continuous grid function for '{name}' returned an "
                    f"array of shape {grid.shape} for the state-choice\n\n"
                    f"{state_choice_dict}\n\nbut every state-choice's grid for "
                    f"'{name}' must be a 1d array of length {expected_length} "
                    f"({length_source}). All state-choices must use grids of the "
                    "same size for a given continuous state; only the grid values "
                    "may vary."
                )
            grid_rows.append(grid)

        grids_per_state_choice[name] = np.stack(grid_rows, axis=0)

    return grids_per_state_choice, resolved_lengths


def _expected_grid_lengths(continuous_states_info):
    expected_lengths = {
        name: len(grid)
        for name, grid in continuous_states_info[
            "additional_continuous_state_grids"
        ].items()
        if grid is not None
    }
    expected_lengths["assets_end_of_period"] = len(
        continuous_states_info["assets_grid_end_of_period"]
    )
    if continuous_states_info.get("assets_begin_of_period") is not None:
        expected_lengths["assets_begin_of_period"] = len(
            continuous_states_info["assets_begin_of_period"]
        )
    return expected_lengths


def check_continuous_grid_consistency_across_shared_children(
    state_choice_space,
    discrete_states_names,
    map_state_choice_to_child_states,
    grids_per_state_choice,
):
    """Check that state-choices sharing a child state agree on their own grid.

    The batch-creation code (``pre_processing/batches/``) deduplicates children purely
    by discrete-state index (``np.unique`` on ``map_state_choice_to_child_states``) and
    computes the continuation-value interpolation once per unique child, reusing it for
    every parent state-choice that maps to it. That shared computation is fed the
    *parent's* own continuous grid (as "last period's grid", see ``law_of_motion.py``).
    If two parent state-choices that share a child disagreed on their own grid, the
    shared computation could only use one of them, silently corrupting the other's
    continuation values.

    Grids live on the state-choice space (see
    ``evaluate_state_specific_continuous_grids`` above), so each row's "own grid" is
    available directly -- no collapsing to the parent's bare state needed, since
    ``map_state_choice_to_child_states`` is already indexed by state-choice row (row
    ``i`` of it is row ``i`` of ``state_choice_space``).

    We reuse the exact same grouping mechanism the batch code uses (``np.unique`` with
    ``return_inverse``) so this check provably guards the invariant that optimization
    depends on, rather than an approximation of it.

    """
    if len(grids_per_state_choice) == 0:
        return

    invalid_idx = np.iinfo(map_state_choice_to_child_states.dtype).max
    n_stochastic_states = map_state_choice_to_child_states.shape[1]

    for name, grid_of_state_choice in grids_per_state_choice.items():
        for k in range(n_stochastic_states):
            child_idx_col = np.asarray(map_state_choice_to_child_states[:, k])
            valid_mask = child_idx_col != invalid_idx

            if not np.any(valid_mask):
                continue

            row_idxs = np.flatnonzero(valid_mask)
            child_idx_valid = child_idx_col[row_idxs]
            own_grid_valid = grid_of_state_choice[row_idxs]

            _, first_occurrence, inverse = np.unique(
                child_idx_valid, return_index=True, return_inverse=True
            )
            representative_grid = own_grid_valid[first_occurrence][inverse]

            mismatch = np.any(own_grid_valid != representative_grid, axis=1)
            if np.any(mismatch):
                bad_local_idx = np.flatnonzero(mismatch)[0]
                bad_row = row_idxs[bad_local_idx]
                rep_row = row_idxs[first_occurrence[inverse[bad_local_idx]]]
                child_state_idx = child_idx_valid[bad_local_idx]

                _raise_grid_mismatch_error(
                    name=name,
                    state_choice_space=state_choice_space,
                    discrete_states_names=discrete_states_names,
                    bad_row=bad_row,
                    rep_row=rep_row,
                    child_state_idx=child_state_idx,
                )


def _raise_grid_mismatch_error(
    name,
    state_choice_space,
    discrete_states_names,
    bad_row,
    rep_row,
    child_state_idx,
):
    def _state_choice_dict(row_idx):
        row = state_choice_space[row_idx]
        example_dict = {key: row[i] for i, key in enumerate(discrete_states_names)}
        example_dict["choice"] = row[-1]
        return example_dict

    raise ValueError(
        f"\n\nThe continuous state '{name}' is declared state-specific via "
        "the continuous_grid_functions argument to create_model_dict, but the "
        "following two state-choice combinations transition to the same child "
        "state "
        f"(child state index {child_state_idx}) while using different grids for "
        f"'{name}':\n\n{_state_choice_dict(bad_row)}\n\nand\n\n"
        f"{_state_choice_dict(rep_row)}\n\n"
        "The grid for a continuous state may only depend on discrete-state "
        "variables and/or the choice, in a way that agrees across every "
        "state-choice combination transitioning to the same child -- e.g. "
        "time-invariant states like sex/education, the period, or a choice that "
        "passes through unchanged (or 1:1) from parent to child. Please revisit "
        f"the grid function for '{name}' or the state space definition."
    )
