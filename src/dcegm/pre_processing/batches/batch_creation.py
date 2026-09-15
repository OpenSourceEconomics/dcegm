import numpy as np

from dcegm.pre_processing.batches.last_two_periods import (
    add_last_two_period_information,
)
from dcegm.pre_processing.batches.single_segment import create_single_segment_of_batches


def _log_segment(id_segment, mode, bool_segment, segment_info, state_choice_space):
    """Print one summary line per segment: period range, batch mode, batch shape."""
    periods = state_choice_space[bool_segment, 0]
    p_lo, p_hi = int(periods.min()), int(periods.max())
    period_str = f"period {p_lo}" if p_lo == p_hi else f"periods {p_lo}-{p_hi}"
    batches = np.asarray(segment_info["batches_state_choice_idx"])
    n_batches, width = batches.shape[0], batches.shape[1]
    leftover = (
        "" if segment_info.get("batches_cover_all", True) else " (+1 leftover batch)"
    )
    # period_max pads every batch to the widest period in the segment.
    pad = ", padded to widest period" if mode == "period_max" else ""
    batch_word = "batch" if n_batches == 1 else "batches"
    print(
        f"    segment {id_segment}: {period_str} [{mode}] -> "
        f"{n_batches} {batch_word} x {width} state-choices{pad}{leftover}"
    )
    # largest_block shrinks the batch size from the segment's last-period size (the
    # search seed) down to the final width; period_max does not search (start is None).
    start = segment_info.get("search_start_size")
    if start is not None:
        print(f"      start search: {start}")
        print(f"      end search:   {width}")


def _log_total(segment_infos):
    """Print the total number of sequential scan steps across all segments."""
    n_seg = segment_infos["n_segments"]
    total = 0
    for s in range(n_seg):
        info = segment_infos[f"batches_info_segment_{s}"]
        n_batches = np.asarray(info["batches_state_choice_idx"]).shape[0]
        total += n_batches + (0 if info.get("batches_cover_all", True) else 1)
    print(f"    {n_seg} segment(s), {total} scan steps total")


def _select_law_of_motion_arrays(
    transition_depends_on_choice,
    child_state_choices_no_proxy,
    rep_parent_state_choice_idx_per_child_state_choice,
    unique_child_states_state_dict,
    rep_parent_state_choice_idx_per_child_state,
    state_row_for_state_choice,
):
    """Keep only the child arrays the taken law-of-motion branch reads.

    Choice-dependent transitions evaluate the law of motion once per child *state-
    choice*, so they need the child's own (non-proxy) state-choice dict plus the per-
    state-choice representative parent; otherwise the coarser per-unique-child-*state*
    dedup suffices (see ``calc_law_of_motion`` in ``law_of_motion.py``). Whichever
    branch is taken is a model-static property
    (``transition_funcs_depend_on_choice["any"]``), so the selection is made once here
    at setup and the unused branch's arrays are never threaded through the backward
    induction.

    ``child_state_choices_no_proxy`` is the transition child (its real state), not the
    proxy value-reuse slot -- the proxy identity stays on the separate
    ``state_choice_mat_child`` entry used to read the stored value/policy.

    """
    if transition_depends_on_choice:
        return {
            "child_state_choices": child_state_choices_no_proxy,
            "rep_parent_state_choice_idx_per_child_state_choice": (
                rep_parent_state_choice_idx_per_child_state_choice
            ),
        }
    return {
        "unique_child_states": unique_child_states_state_dict,
        "rep_parent_state_choice_idx_per_child_state": (
            rep_parent_state_choice_idx_per_child_state
        ),
        "state_row_for_state_choice": state_row_for_state_choice,
    }


def _bundle_segment_law_of_motion_arrays(segment_info, transition_depends_on_choice):
    """Fold a scan segment's per-branch law-of-motion arrays into one dict.

    Pops the four alternatives out of ``segment_info`` (and its leftover
    ``last_batch_info``, if any) and replaces them with a single
    ``law_of_motion_arrays`` entry carrying only the taken branch -- the dict
    ``calc_law_of_motion`` reads, threaded through the backward-induction scan.

    """
    for info in (segment_info, segment_info.get("last_batch_info")):
        if info is None:
            continue
        info["law_of_motion_arrays"] = _select_law_of_motion_arrays(
            transition_depends_on_choice=transition_depends_on_choice,
            child_state_choices_no_proxy=info.pop("state_choices_childs_no_proxy"),
            rep_parent_state_choice_idx_per_child_state_choice=info.pop(
                "rep_parent_state_choice_idx_per_child_state_choice"
            ),
            unique_child_states_state_dict=info.pop(
                "state_choices_unique_child_states"
            ),
            rep_parent_state_choice_idx_per_child_state=info.pop(
                "rep_parent_state_choice_idx_per_child_state"
            ),
            state_row_for_state_choice=info.pop("state_row_for_state_choice"),
        )


def _bundle_final_period_law_of_motion_arrays(
    last_two_period_info, transition_depends_on_choice, state_space_dict
):
    """Fold the final period's per-branch law-of-motion arrays into one dict.

    The final period reaches ``calc_law_of_motion`` outside the scan (see
    ``final_periods.solve_final_period``), so it gets the same ``law_of_motion_arrays``
    bundle, just from the final-period-specific source keys. On the state-dedup branch
    the unique child *states* are gathered into a state dict here (mirroring the per-
    batch gather in ``single_segment.py``) rather than in the solve.

    """
    unique_final_period_states = last_two_period_info.pop("unique_final_period_states")
    rep_parent_idx_per_state_choice = last_two_period_info.pop(
        "rep_sec_last_period_parent_idx_per_final_state_choice"
    )
    rep_parent_idx_per_state = last_two_period_info.pop(
        "representative_second_last_period_parent_idx_per_final_state"
    )
    state_row_for_state_choice = last_two_period_info.pop(
        "state_row_for_final_period_state_choice"
    )
    unique_child_states_state_dict = {
        key: var[unique_final_period_states] for key, var in state_space_dict.items()
    }
    # The final period's children are its own (solved, real) state-choices, so the
    # transition child is simply state_choice_mat_final_period -- no proxy is
    # involved here (death proxies target the last period, which is solved).
    last_two_period_info["law_of_motion_arrays"] = _select_law_of_motion_arrays(
        transition_depends_on_choice=transition_depends_on_choice,
        child_state_choices_no_proxy=last_two_period_info[
            "state_choice_mat_final_period"
        ],
        rep_parent_state_choice_idx_per_child_state_choice=rep_parent_idx_per_state_choice,
        unique_child_states_state_dict=unique_child_states_state_dict,
        rep_parent_state_choice_idx_per_child_state=rep_parent_idx_per_state,
        state_row_for_state_choice=state_row_for_state_choice,
    )


def bundle_law_of_motion_arrays(
    batch_info, transition_depends_on_choice, state_space_dict
):
    """Assemble every ``law_of_motion_arrays`` bundle in a finished ``batch_info``.

    Run once at model setup: walks the final-period info and each scan segment
    (including any leftover batch) and replaces the per-branch law-of-motion index
    arrays with the single dict ``calc_law_of_motion`` reads, dropping the branch
    that this model never takes.

    """
    _bundle_final_period_law_of_motion_arrays(
        batch_info["last_two_period_info"],
        transition_depends_on_choice=transition_depends_on_choice,
        state_space_dict=state_space_dict,
    )
    if batch_info["two_period_model"]:
        return batch_info
    for id_segment in range(batch_info["n_segments"]):
        _bundle_segment_law_of_motion_arrays(
            batch_info[f"batches_info_segment_{id_segment}"],
            transition_depends_on_choice=transition_depends_on_choice,
        )
    return batch_info


def create_batches_and_information(
    model_structure,
    n_periods,
    min_period_batch_segments,
    batch_mode,
):
    """Batches are used instead of periods to have chunks of equal sized state choices.
    The returned batch information dictionary contains the following arrays
    reflecting steps in the backward induction:

        - batches_state_choice_idx: The state choice indexes in each batch to be solved.
            To solve the state choices in the egm step, we have to look at the child states
            and the corresponding state choice indexes in the child states. For that we save
            the following:
        - child_state_choice_idxs_to_interp: The state choice indexes in we need to
            interpolate the wealth on.
        - child_states_idxs: The parent state indexes of the child states, i.e. the
            child states themself. We calculate the resources at the beginning of
            period before the backwards induction with the budget equation for each
            saving and income shock grid point.

        Note: These two index arrays containing indexes on the whole
        state/state-choice space.

    Once we have the interpolated in all possible child state-choice states,
    we rearange them to an array with row as states and columns as choices to
    aggregate over the choices. This is saved in:

        - child_state_choices_to_aggr_choice: The state choice indexes in the child
            states to aggregate over. Note these are relative indexes indexing to the
            batch arrays from the step before.
    Now we have for each child state a value/marginal utility with the index arrays
    above and what is missing is the mapping for the exogenous/stochastic processes.
    This is saved via:
        - child_states_to_integrate_exog: The state choice indexes in the child states
            to integrate over the exogenous processes. This is a relative index to the
            batch arrays from the step before.

    """
    last_two_period_info = add_last_two_period_information(
        n_periods=n_periods,
        model_structure=model_structure,
    )

    if n_periods == 2:
        # In the case of a two period model, we just need the information of the last
        # two periods
        batch_info = {
            "two_period_model": True,
            "last_two_period_info": last_two_period_info,
        }

        return batch_info

    state_choice_space = model_structure["state_choice_space"]
    bool_state_choices_to_batch = state_choice_space[:, 0] < n_periods - 2
    print("  batches:")

    valid_batch_modes = {"largest_block", "period_max"}

    if min_period_batch_segments is None:
        if isinstance(batch_mode, list):
            raise ValueError(
                "If min_period_batch_segments is not supplied, batch_mode must be a string."
            )
        if batch_mode not in valid_batch_modes:
            raise ValueError(
                f"batch_mode must be one of {valid_batch_modes}. Got {batch_mode}."
            )

        single_batch_segment_info = create_single_segment_of_batches(
            bool_state_choices_to_batch,
            model_structure,
            batch_mode=batch_mode,
        )
        segment_infos = {
            "n_segments": 1,
            "batches_info_segment_0": single_batch_segment_info,
        }
        _log_segment(
            0,
            batch_mode,
            bool_state_choices_to_batch,
            single_batch_segment_info,
            state_choice_space,
        )

    else:

        if isinstance(min_period_batch_segments, int):
            n_segments = 2
            min_period_batch_segments = [min_period_batch_segments]
        elif isinstance(min_period_batch_segments, list):
            n_segments = len(min_period_batch_segments) + 1
        else:
            raise ValueError("So far only int or list separation is supported.")

        # Check if periods are increasing and at least two periods apart.
        # Also that they are at least two periods smaller than n_periods - 2
        if not all(
            min_period_batch_segments[i] < min_period_batch_segments[i + 1]
            for i in range(len(min_period_batch_segments) - 1)
        ) or not all(
            min_period_batch_segments[i] < n_periods - 2 - 2
            for i in range(len(min_period_batch_segments))
        ):
            raise ValueError(
                "The periods to split the batches have to be increasing and at least two periods apart."
            )

        if isinstance(batch_mode, str):
            if batch_mode not in valid_batch_modes:
                raise ValueError(
                    f"batch_mode must be one of {valid_batch_modes}. Got {batch_mode}."
                )
            batch_mode = [batch_mode] * n_segments
        elif isinstance(batch_mode, list):
            if len(batch_mode) != n_segments:
                raise ValueError(
                    "If min_period_batch_segments is supplied, batch_mode must be a list with one entry per segment."
                )
            if not all(mode in valid_batch_modes for mode in batch_mode):
                raise ValueError(
                    f"All entries in batch_mode must be one of {valid_batch_modes}."
                )
        else:
            raise ValueError("batch_mode must be a string or a list of strings.")

        segment_infos = {
            "n_segments": n_segments,
        }

        for id_segment in range(n_segments - 1):

            # Start from the end and assign segments, i.e. segment 0 starts at
            # min_periods_to_split[-1] and ends at n_periods - 2
            period_to_split = min_period_batch_segments[-id_segment - 1]

            split_cond = state_choice_space[:, 0] < period_to_split
            bool_state_choices_segment = bool_state_choices_to_batch & (~split_cond)

            segment_batch_info = create_single_segment_of_batches(
                bool_state_choices_segment,
                model_structure,
                batch_mode=batch_mode[id_segment],
            )
            segment_infos[f"batches_info_segment_{id_segment}"] = segment_batch_info
            _log_segment(
                id_segment,
                batch_mode[id_segment],
                bool_state_choices_segment,
                segment_batch_info,
                state_choice_space,
            )

            # Set the bools to False which have been batched already
            bool_state_choices_to_batch = bool_state_choices_to_batch & split_cond

        last_segment_batch_info = create_single_segment_of_batches(
            bool_state_choices_to_batch,
            model_structure,
            batch_mode=batch_mode[n_segments - 1],
        )

        # We loop until n_segments - 2 and then add the last segment
        segment_infos[f"batches_info_segment_{n_segments - 1}"] = (
            last_segment_batch_info
        )
        _log_segment(
            n_segments - 1,
            batch_mode[n_segments - 1],
            bool_state_choices_to_batch,
            last_segment_batch_info,
            state_choice_space,
        )

    _log_total(segment_infos)

    batch_info = {
        # First two bools determining the structure of solution functions we call
        "two_period_model": False,
        **segment_infos,
        "last_two_period_info": last_two_period_info,
    }

    return batch_info
