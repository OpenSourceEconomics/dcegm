"""State-specific continuous state grids.

See docs/source/development/internals/state_specific_continuous_grids_plan.md for
the design and the reasoning behind it.

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

    Also validates that every state-choice's grid has the length declared in
    ``model_config["continuous_states"]`` for that name. Storage containers are
    rectangular (same number of grid points for every state-choice, only the
    values differ -- see the implementation plan), so a mismatched length would
    otherwise silently corrupt shapes downstream instead of failing here, close to
    the user-supplied function that caused it.

    This is a one-time NumPy-side pass over the full state-choice space, done once
    during model-structure construction. The result is used to build the
    child-sharing consistency check (see
    ``check_continuous_grid_consistency_across_shared_children`` below) and is not
    stored in ``model_structure`` or threaded through solving.

    """
    if len(state_specific_names) == 0:
        return {}

    expected_lengths = _expected_grid_lengths(continuous_states_info)

    grids_per_state_choice = {}
    for name in state_specific_names:
        grid_func = continuous_grid_functions[name]
        expected_length = expected_lengths[name]

        grid_rows = []
        for row in state_choice_space:
            state_choice_dict = {
                key: row[i] for i, key in enumerate(discrete_state_choice_names)
            }
            grid = np.asarray(grid_func(**state_choice_dict))
            if grid.ndim != 1 or grid.shape[0] != expected_length:
                raise ValueError(
                    f"\n\nThe continuous grid function for '{name}' returned an "
                    f"array of shape {grid.shape} for the state-choice\n\n"
                    f"{state_choice_dict}\n\nbut every state-choice's grid for "
                    f"'{name}' must be a 1d array of length {expected_length} "
                    "(the length declared in "
                    f"model_config['continuous_states']['{name}']). All "
                    "state-choices must use grids of the same size for a given "
                    "continuous state; only the grid values may vary."
                )
            grid_rows.append(grid)

        grids_per_state_choice[name] = np.stack(grid_rows, axis=0)

    return grids_per_state_choice


def _expected_grid_lengths(continuous_states_info):
    expected_lengths = {
        name: len(grid)
        for name, grid in continuous_states_info[
            "additional_continuous_state_grids"
        ].items()
    }
    expected_lengths["assets_end_of_period"] = len(
        continuous_states_info["assets_grid_end_of_period"]
    )
    if "assets_begin_of_period" in continuous_states_info:
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

    The batch-creation code (``pre_processing/batches/``) deduplicates children
    purely by discrete-state index (``np.unique`` on
    ``map_state_choice_to_child_states``) and computes the continuation-value
    interpolation once per unique child, reusing it for every parent state-choice
    that maps to it. That shared computation is fed the *parent's* own continuous
    grid (as "last period's grid", see ``law_of_motion.py``). If two parent
    state-choices that share a child disagreed on their own grid, the shared
    computation could only use one of them, silently corrupting the other's
    continuation values.

    Grids live on the state-choice space (see
    ``evaluate_state_specific_continuous_grids`` above), so each row's "own grid"
    is available directly -- no collapsing to the parent's bare state needed, since
    ``map_state_choice_to_child_states`` is already indexed by state-choice row
    (row ``i`` of it is row ``i`` of ``state_choice_space``).

    We reuse the exact same grouping mechanism the batch code uses
    (``np.unique`` with ``return_inverse``) so this check provably guards the
    invariant that optimization depends on, rather than an approximation of it.
    See docs/source/development/internals/state_specific_continuous_grids_plan.md.

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
