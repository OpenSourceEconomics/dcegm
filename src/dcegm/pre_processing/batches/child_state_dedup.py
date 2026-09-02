import numpy as np


def compute_child_dedup_for_batch(
    batch,
    map_state_choice_to_child_states,
    map_state_choice_to_index,
    state_space,
    n_state_vars,
    invalid_state_idx,
    out_of_bounds_state_choice_idx,
):
    """Deduplicate a batch's child states/state-choices; find a representative parent.

    Shared by both batch-size algorithms (``algo_batch_size.py``'s "largest_block"
    and ``single_segment.py``'s "period_max") -- they deduplicated this identically
    before this was factored out, so keeping one implementation avoids having to fix
    the same bug twice.

    The batch code computes the continuation-value interpolation once per unique
    child (deduplicated purely by discrete-state index, see the two ``np.unique``
    calls below) and reuses it for every parent state-choice that maps to it. That
    computation needs the *parent's* own continuous grid as input to the child's
    law-of-motion function (see ``law_of_motion.py``) -- not the child's. Since
    the computation runs once per unique child rather than once per parent, it
    needs *a* representative parent for each unique child; any
    one of them works, because
    ``check_continuous_grid_consistency_across_shared_children`` (run once at
    model-build time, before batching) already guarantees every parent state-choice
    sharing a child agrees on its own grid.

    The representative parent is a *state-choice* (an index into
    ``batch``/``state_choice_space``), not a bare state: grids live on the
    state-choice space (that's where the solution itself lives), so ``batch``
    already gives us exactly the identity we need, with no extra lookup required.

    Returns:
        tuple:

        - child_states_to_integrate_exog (np.ndarray): shape (len(batch), n_stochastic),
            maps each (parent row, stochastic draw) to a position in the unique
            child *state* space.
        - child_state_choices_to_aggr_choice (np.ndarray): shape (n_unique_child_states,
            n_choices), maps each (child state, choice) to a position in the unique
            child *state-choice* space (or an out-of-bounds sentinel).
        - unique_child_state_choice_idxs (np.ndarray): the deduplicated child
            state-choice indices themselves.
        - representative_parent_state_choice_for_child (np.ndarray): same length as
            unique_child_state_choice_idxs; for each unique child state-choice, the
            state-choice index of one parent (from this batch) that transitions to
            it.

    """
    child_states_idxs = map_state_choice_to_child_states[batch]
    n_stochastic_states = child_states_idxs.shape[1]

    unique_child_states, first_occurrence_state, inverse_ids = np.unique(
        child_states_idxs, return_index=True, return_inverse=True
    )
    child_states_to_integrate_exog = inverse_ids.reshape(child_states_idxs.shape)

    # A representative parent state-choice (by local row within `batch`) for each
    # unique child state -- the batch row/stochastic-draw where that child state
    # first occurs.
    representative_parent_row = first_occurrence_state // n_stochastic_states
    representative_parent_state_choice_per_child_state = batch[
        representative_parent_row
    ]

    child_states_batch = np.take(state_space, unique_child_states, axis=0)
    child_states_tuple = tuple(child_states_batch[:, i] for i in range(n_state_vars))
    unique_state_choice_idxs_childs = map_state_choice_to_index[child_states_tuple]

    # The representative parent is a property of the child *state*, not of which of
    # its choices we're looking at, so it is the same across all n_choices columns
    # for a given child-state row.
    representative_parent_state_choice_per_child_choice = np.broadcast_to(
        representative_parent_state_choice_per_child_state[:, None],
        unique_state_choice_idxs_childs.shape,
    )

    (
        unique_child_state_choice_idxs,
        first_occurrence_state_choice,
        inverse_child_state_choice_ids,
    ) = np.unique(
        unique_state_choice_idxs_childs, return_index=True, return_inverse=True
    )
    representative_parent_state_choice_for_child = (
        representative_parent_state_choice_per_child_choice.ravel()[
            first_occurrence_state_choice
        ]
    )

    if (
        len(unique_child_state_choice_idxs) > 0
        and unique_child_state_choice_idxs[-1] == invalid_state_idx
    ):
        unique_child_state_choice_idxs = unique_child_state_choice_idxs[:-1]
        representative_parent_state_choice_for_child = (
            representative_parent_state_choice_for_child[:-1]
        )
        inverse_child_state_choice_ids[
            inverse_child_state_choice_ids >= np.max(inverse_child_state_choice_ids)
        ] = out_of_bounds_state_choice_idx

    child_state_choices_to_aggr_choice = inverse_child_state_choice_ids.reshape(
        unique_state_choice_idxs_childs.shape
    )

    return (
        child_states_to_integrate_exog,
        child_state_choices_to_aggr_choice,
        unique_child_state_choice_idxs,
        representative_parent_state_choice_for_child,
    )
