import numpy as np

from dcegm.pre_processing.model_structure.shared import span_subspace
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options


def process_endog_state_specifications(state_space_options, model_params):
    """Get number of endog states which we loop over when creating the state space."""

    def dummy_sparsity_func(**kwargs):
        return True

    if state_space_options.get("endogenous_states"):

        endog_state_keys = state_space_options["endogenous_states"].keys()

        if "sparsity_condition" in state_space_options["endogenous_states"].keys():
            endog_states_names = list(set(endog_state_keys) - {"sparsity_condition"})
            sparsity_func = determine_function_arguments_and_partial_options(
                func=state_space_options["endogenous_states"]["sparsity_condition"],
                options=model_params,
            )
        else:
            sparsity_func = dummy_sparsity_func
            endog_states_names = list(endog_state_keys)

        endog_state_space = span_subspace(
            subdict_of_space=state_space_options["endogenous_states"],
            states_names=endog_states_names,
        )

    else:
        endog_states_names = []

        endog_state_space = np.array([[0]])
        sparsity_func = dummy_sparsity_func

    return (
        endog_state_space,
        endog_states_names,
        sparsity_func,
    )
