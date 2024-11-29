from dcegm.pre_processing.model_structure.shared import span_subspace
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options


def process_endog_state_specifications(state_space_options, model_params):
    """Get number of endog states which we loop over when creating the state space."""
    if state_space_options.get("endogenous_states"):

        endog_state_keys = state_space_options["endogenous_states"].keys()

        if "sparsity_condition" in state_space_options["endogenous_states"].keys():
            endog_states_names = list(set(endog_state_keys) - {"sparsity_condition"})
            sparsity_cond_specified = True
        else:
            sparsity_cond_specified = False
            endog_states_names = list(endog_state_keys)

        endog_state_space = span_subspace(
            subdict_of_space=state_space_options["endogenous_states"],
            states_names=endog_states_names,
        )
        n_endog_states = endog_state_space.shape[0]

    else:
        endog_states_names = []
        n_endog_states = 1

        endog_state_space = None
        sparsity_cond_specified = False

    sparsity_func = select_sparsity_function(
        sparsity_cond_specified=sparsity_cond_specified,
        state_space_options=state_space_options,
        model_params=model_params,
    )

    endog_states_add_func = create_endog_state_add_function(endog_state_space)

    return (
        endog_states_add_func,
        endog_states_names,
        n_endog_states,
        sparsity_func,
    )


def select_sparsity_function(
    sparsity_cond_specified, state_space_options, model_params
):
    if sparsity_cond_specified:
        sparsity_func = determine_function_arguments_and_partial_options(
            func=state_space_options["endogenous_states"]["sparsity_condition"],
            options=model_params,
        )
    else:

        def sparsity_func(**kwargs):
            return True

    return sparsity_func


def create_endog_state_add_function(endog_state_space):
    if endog_state_space is None:

        def add_endog_states(id_endog_state):
            return []

    else:

        def add_endog_states(id_endog_state):
            return list(endog_state_space[id_endog_state])

    return add_endog_states
