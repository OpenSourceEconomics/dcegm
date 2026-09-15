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

    The proxy (``map_state_choice_to_index`` here is the *with-proxy* indexer) is a
    pure value-reuse pointer, kept out of the transition granularity: the rows are
    the *non-proxy* child state-choices, so distinct children that reuse one solved
    slot (death at different ages -> one last-period slot) stay separate for the law
    of motion, and only the value/policy lookup follows the proxy (``value_slot``,
    which then repeats across those rows).

    Also exposes the *state*-level dedup: ``unique_child_states``,
    ``representative_parent_state_choice_per_child_state``, and
    ``state_row_for_state_choice``. These let a caller evaluate something once per
    unique child *state* (e.g. a law-of-motion function that doesn't depend on the
    child's own choice) and then ``np.take``/``jnp.take`` the result out to the
    state-choice granularity ``value_slot`` needs for reading stored policy/value --
    mirrors the existing ``child_states_to_integrate_exog`` gather used one stage
    later, in ``calculate_candidate_solutions_from_euler_equation``.

    Returns:
        tuple:

        - child_states_to_integrate_exog (np.ndarray): shape (len(batch), n_stochastic),
            maps each (parent row, stochastic draw) to a position in the unique
            child *state* space.
        - child_state_choices_to_aggr_choice (np.ndarray): shape (n_unique_child_states,
            n_choices), maps each (child state, choice) to its row in the non-proxy
            child *state-choice* axis (or an out-of-bounds sentinel).
        - value_slot (np.ndarray): one entry per non-proxy child state-choice; the
            solved (proxy) state-choice index whose value/policy/endog that row
            reads. Repeats across rows whose children share a proxy slot.
        - representative_parent_state_choice_for_child (np.ndarray): same length as
            value_slot; for each row, the state-choice index of one parent (from
            this batch) that transitions to it.
        - unique_child_states (np.ndarray): the deduplicated child *state* indices
            (into ``state_space``) -- one entry per unique child state, collapsing
            across that state's own choices.
        - representative_parent_state_choice_per_child_state (np.ndarray): same
            length as unique_child_states; for each unique child state, the
            state-choice index of one parent that transitions to it (same value
            used across all of that state's own choices in
            representative_parent_state_choice_for_child).
        - state_row_for_state_choice (np.ndarray): same length as value_slot; for
            each non-proxy child state-choice, its row position in
            unique_child_states -- the gather index needed to expand a per-state
            result out to per-state-choice granularity.

    """
    child_states_idxs = map_state_choice_to_child_states[batch]
    n_stochastic_states = child_states_idxs.shape[1]

    unique_child_states, first_occurrence_state, inverse_ids = np.unique(
        child_states_idxs, return_index=True, return_inverse=True
    )
    child_states_to_integrate_exog = inverse_ids.reshape(child_states_idxs.shape)
    n_unique_child_states = unique_child_states.shape[0]

    # A representative parent state-choice (by local row within `batch`) for each
    # unique child state -- the batch row/stochastic-draw where that child state
    # first occurs.
    representative_parent_row = first_occurrence_state // n_stochastic_states
    representative_parent_state_choice_per_child_state = batch[
        representative_parent_row
    ]

    child_states_batch = np.take(state_space, unique_child_states, axis=0)
    child_states_tuple = tuple(child_states_batch[:, i] for i in range(n_state_vars))
    # Proxy solved state-choice index for each (unique child state, choice) cell.
    # `map_state_choice_to_index` is the *with-proxy* indexer, so a child whose own
    # solution is reused points at the reused (proxy) slot; cells the child does not
    # admit map to `invalid_state_idx`.
    proxy_state_choice_idxs_childs = map_state_choice_to_index[child_states_tuple]

    # The representative parent is a property of the child *state*, not of which of
    # its choices we're looking at, so it is the same across all n_choices columns
    # for a given child-state row.
    representative_parent_per_cell = np.broadcast_to(
        representative_parent_state_choice_per_child_state[:, None],
        proxy_state_choice_idxs_childs.shape,
    )
    state_row_per_cell = np.broadcast_to(
        np.arange(n_unique_child_states)[:, None],
        proxy_state_choice_idxs_childs.shape,
    )

    # One row per (child state, choice) the child admits -- the *non-proxy*
    # granularity the law of motion needs. Distinct children that reuse the same
    # solved slot (e.g. death at different ages, all proxied to one last-period
    # slot) stay separate rows here, because the transition *into* them differs;
    # only the value/policy lookup collapses, via the repeated ``value_slot`` below.
    # Sorting the admitted cells by their proxy solved index reproduces the previous
    # ``np.unique`` ordering bit-for-bit whenever no proxy actually collapses two
    # children (proxy == actual) -- i.e. every model without a cross-period proxy.
    flat_proxy_idx = proxy_state_choice_idxs_childs.ravel()
    admitted_cell = flat_proxy_idx != invalid_state_idx
    ordered_cells = np.where(admitted_cell)[0][
        np.argsort(flat_proxy_idx[admitted_cell], kind="stable")
    ]

    # value_slot: the solved (proxy) state-choice whose value/policy/endog this row
    # reads -- repeated across rows that share a proxy. state_row_for_state_choice:
    # the row's own (non-proxy) child *state* row, for expanding a per-state law of
    # motion back out to per-state-choice granularity.
    value_slot = flat_proxy_idx[ordered_cells]
    state_row_for_state_choice = state_row_per_cell.ravel()[ordered_cells]
    representative_parent_state_choice_for_child = (
        representative_parent_per_cell.ravel()[ordered_cells]
    )

    # For each (child state, choice) cell, its row position, or an out-of-bounds
    # sentinel for choices the child does not admit -- the choice-aggregation map.
    flat_aggr = np.full(
        flat_proxy_idx.shape[0],
        fill_value=out_of_bounds_state_choice_idx,
        dtype=int,
    )
    flat_aggr[ordered_cells] = np.arange(ordered_cells.shape[0])
    child_state_choices_to_aggr_choice = flat_aggr.reshape(
        proxy_state_choice_idxs_childs.shape
    )

    return (
        child_states_to_integrate_exog,
        child_state_choices_to_aggr_choice,
        value_slot,
        representative_parent_state_choice_for_child,
        unique_child_states,
        representative_parent_state_choice_per_child_state,
        state_row_for_state_choice,
    )
