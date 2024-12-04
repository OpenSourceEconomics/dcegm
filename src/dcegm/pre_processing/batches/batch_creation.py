from dcegm.pre_processing.batches.last_two_periods import (
    add_last_two_period_information,
)
from dcegm.pre_processing.batches.single_segment import create_single_segment_of_batches


def create_batches_and_information(
    model_structure,
    state_space_options,
):
    """Batches are used instead of periods to have chunks of equal sized state choices.
    The batch information dictionary contains the following arrays reflecting the.

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

    n_periods = state_space_options["n_periods"]

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

    # if "split_model_calc" not in state_space_options.keys():

    state_choice_space = model_structure["state_choice_space"]
    idx_state_choices_to_batch = state_choice_space[:, 0] < n_periods - 2

    single_batch_segment_info = create_single_segment_of_batches(
        idx_state_choices_to_batch, model_structure
    )

    batch_info = {
        # First two bools determining the structure of solution functions we call
        "two_period_model": False,
        **single_batch_segment_info,
        "last_two_period_info": last_two_period_info,
    }

    return batch_info
