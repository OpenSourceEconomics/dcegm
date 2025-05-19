import numpy as np

from dcegm.pre_processing.model_structure.shared import span_subspace
from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)


def process_endog_state_specifications(state_space_options):
    """Get number of endog states which we loop over when creating the state space."""

    if state_space_options.get("deterministic_states"):

        endog_states_names = list(state_space_options["deterministic_states"].keys())
        endog_state_space = span_subspace(
            subdict_of_space=state_space_options["deterministic_states"],
            states_names=endog_states_names,
        )

    else:
        endog_states_names = []
        endog_state_space = np.array([[0]])

    return (
        endog_state_space,
        endog_states_names,
    )
