import numpy as np

from dcegm.pre_processing.batches.last_two_periods import (
    add_last_two_period_information,
)
from dcegm.pre_processing.batches.single_segment import create_single_segment_of_batches


def create_batches_and_information(
    model_structure,
    n_periods,
    min_period_batch_segments=None,
):
    """Batches are used instead of periods to have chunks of equal sized state choices.
    The batch inparams=paramsformation dictionary contains the following arrays
    reflecting the.

    steps in the backward induction:
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

    if min_period_batch_segments is None:

        single_batch_segment_info = create_single_segment_of_batches(
            bool_state_choices_to_batch, model_structure
        )
        segment_infos = {
            "n_segments": 1,
            "batches_info_segment_0": single_batch_segment_info,
        }

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
                bool_state_choices_segment, model_structure
            )
            segment_infos[f"batches_info_segment_{id_segment}"] = segment_batch_info

            # Set the bools to False which have been batched already
            bool_state_choices_to_batch = bool_state_choices_to_batch & split_cond

        last_segment_batch_info = create_single_segment_of_batches(
            bool_state_choices_to_batch, model_structure
        )

        # We loop until n_segments - 2 and then add the last segment
        segment_infos[f"batches_info_segment_{n_segments - 1}"] = (
            last_segment_batch_info
        )

    batch_info = {
        # First two bools determining the structure of solution functions we call
        "two_period_model": False,
        **segment_infos,
        "last_two_period_info": last_two_period_info,
    }

    return batch_info
