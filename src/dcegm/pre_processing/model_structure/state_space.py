"""Functions for creating internal state space objects."""

import numpy as np
import pandas as pd

from dcegm.pre_processing.model_structure.deterministic_states import (
    process_endog_state_specifications,
)
from dcegm.pre_processing.model_structure.shared import create_indexer_for_space
from dcegm.pre_processing.model_structure.stochastic_states import (
    process_stochastic_model_specifications,
)
from dcegm.pre_processing.shared import create_array_with_smallest_int_dtype


def create_state_space(model_config, sparsity_condition, debugging=False):
    """Create state space object and indexer.

    We need to add the convention for the state space objects.

    Args:
        options (dict): Options dictionary.

    Returns:
        Dict:

        - state_vars (list): List of state variables.
        - state_space (np.ndarray): 2d array of shape (n_states, n_state_variables + 1)
            which serves as a collection of all possible states. By convention,
            the first column must contain the period and the last column the
            exogenous processes. Any other state variables are in between.
            E.g. if the two state variables are period and lagged choice and all choices
            are admissible in each period, the shape of the state space array is
            (n_periods * n_choices, 3).
        - map_state_to_index (np.ndarray): Indexer array that maps states to indexes.
            The shape of this object is quite complicated. For each state variable it
            has the number of possible states as rows, i.e.
            (n_poss_states_state_var_1, n_poss_states_state_var_2, ....).

    """
    n_periods = model_config["n_periods"]
    n_choices = len(model_config["choices"])

    (
        endog_state_space,
        endog_states_names,
    ) = process_endog_state_specifications(state_space_options=model_config)
    state_names_without_stochastic = ["period", "lagged_choice"] + endog_states_names

    (
        stochastic_state_names,
        stochastic_state_space_raw,
    ) = process_stochastic_model_specifications(model_config=model_config)
    discrete_states_names = state_names_without_stochastic + stochastic_state_names

    n_stochastic_states = stochastic_state_space_raw.shape[0]

    state_space_list = []
    list_of_states_proxied_from = []
    list_of_states_proxied_to = []
    proxies_exist = False

    # For debugging we create some additional containers
    full_state_space_list = []
    proxy_list = []
    valid_list = []

    for period in range(n_periods):
        for endog_state_id in range(endog_state_space.shape[0]):
            for lagged_choice in range(n_choices):
                # Select the endogenous state, if present
                if len(endog_states_names) == 0:
                    endog_states = []
                else:
                    endog_states = list(endog_state_space[endog_state_id])

                for stochastic_state_id in range(n_stochastic_states):
                    stochastic_states = stochastic_state_space_raw[
                        stochastic_state_id, :
                    ]

                    # Create the state vector
                    state = (
                        [period, lagged_choice] + endog_states + list(stochastic_states)
                    )

                    full_state_space_list += [state]

                    # Transform to dictionary to call sparsity function from user
                    state_dict = {
                        discrete_states_names[i]: state_value
                        for i, state_value in enumerate(state)
                    }

                    # Check if the state is valid by calling the sparsity function
                    sparsity_output = sparsity_condition(**state_dict)

                    # The sparsity condition can either return a boolean indicating if the state
                    # is valid or not, or a dictionary which contains the valid state which is used
                    # instead as a child state for other states. If a state is invalid because of the
                    # stochastic state component, the user must specify a valid state to use instead, as
                    # we assume a state choice combination has n_stochastic_states children.
                    # We do check later if the user correctly specified the proxy state. Here we just check
                    # the format of the output. To simplify this specification the user can also return the same
                    # state as used as input. Then the state is just valid. This allows to easier define a proxy
                    # state for a whole set of states.
                    if isinstance(sparsity_output, dict):
                        # Check if dictionary keys are the same
                        if set(sparsity_output.keys()) != set(discrete_states_names):
                            raise ValueError(
                                f" The state \n\n{sparsity_output}\n\n returned by the sparsity condition "
                                f"does not have the correct format. The dictionary keys should be the same as "
                                f"the discrete state names defined in the state space options. These are"
                                f": \n\n{discrete_states_names}\n\n."
                            )

                        # Check if each value is integer or array with dtype int
                        for key, value in sparsity_output.items():
                            if isinstance(value, int) or np.issubdtype(
                                value.dtype, np.integer
                            ):
                                pass
                            else:
                                raise ValueError(
                                    f"The value of the key {key} in the state \n\n{sparsity_output}\n\n"
                                    f"returned by the sparsity condition is not of integer type."
                                )

                        # Now check if the state is actually the same as the input state
                        is_same_state = True
                        for key, value in sparsity_output.items():
                            same_value = state_dict[key] == value
                            is_same_state &= same_value

                        if is_same_state:
                            state_is_valid = True
                            proxy_list += [False]
                        else:
                            proxy_list += [True]
                            state_is_valid = False
                            proxies_exist = True
                            list_of_states_proxied_from += [state]
                            state_list_proxied_to = [
                                sparsity_output[key] for key in discrete_states_names
                            ]
                            list_of_states_proxied_to += [state_list_proxied_to]
                    elif isinstance(sparsity_output, bool):
                        state_is_valid = sparsity_output
                        proxy_list += [False]
                    else:
                        raise ValueError(
                            f"The sparsity condition for the state \n\n{state_dict}\n\n"
                            f"returned an output of the wrong type. It should return either a boolean"
                            f"or a dictionary."
                        )

                    valid_list += [state_is_valid]
                    if state_is_valid:
                        state_space_list += [state]

    # Generate state space including proxies and max values
    state_space_full = np.array(full_state_space_list)
    max_values_unrestricted = np.max(state_space_full, axis=0)
    proxy_or_valid = np.array(valid_list) | np.array(proxy_list)
    state_space_incl_proxies = state_space_full[proxy_or_valid]

    state_space_raw = np.array(state_space_list)
    state_space = create_array_with_smallest_int_dtype(state_space_raw)
    map_state_to_index, invalid_index = create_indexer_for_space(
        state_space, max_var_values=max_values_unrestricted
    )

    if proxies_exist:
        # If proxies exist we create a different indexer, to map
        # the child states of state choices later to proxied states
        map_state_to_index_with_proxies = create_indexer_inclucing_proxies(
            map_state_to_index,
            list_of_states_proxied_from,
            list_of_states_proxied_to,
            discrete_states_names,
            invalid_index,
        )
        map_state_to_index_with_proxy = map_state_to_index_with_proxies
    else:
        map_state_to_index_with_proxy = map_state_to_index

    state_space_dict = {
        key: create_array_with_smallest_int_dtype(state_space[:, i])
        for i, key in enumerate(discrete_states_names)
    }

    stochastic_state_space = create_array_with_smallest_int_dtype(
        stochastic_state_space_raw
    )

    dict_of_state_space_objects = {
        "state_space_incl_proxies": state_space_incl_proxies,
        "state_space": state_space,
        "state_space_dict": state_space_dict,
        "map_state_to_index": map_state_to_index,
        "map_state_to_index_with_proxy": map_state_to_index_with_proxy,
        "stochastic_state_space": stochastic_state_space,
        "stochastic_states_names": stochastic_state_names,
        "state_names_without_stochastic": state_names_without_stochastic,
        "discrete_states_names": discrete_states_names,
    }

    # If debugging is called we create a dataframe with detailed information on
    # full state space
    if debugging:
        debug_df = pd.DataFrame(data=state_space_full, columns=discrete_states_names)
        debug_df["is_valid"] = valid_list
        debug_df["is_proxied"] = proxy_list

        if proxies_exist:
            array_of_states_proxied_to = np.array(list_of_states_proxied_to)
            tuple_of_states_proxied_from = tuple(
                array_of_states_proxied_to[:, i]
                for i in range(array_of_states_proxied_to.shape[1])
            )
            full_indexer, _ = create_indexer_for_space(state_space_full)
            idxs_proxied_to = full_indexer[tuple_of_states_proxied_from]
            debug_df["idxs_proxied_to"] = -9999
            debug_df.loc[debug_df["is_proxied"], "idxs_proxied_to"] = idxs_proxied_to

        return debug_df

    return dict_of_state_space_objects


def create_indexer_inclucing_proxies(
    map_state_to_index,
    list_of_states_proxied_from,
    list_of_states_proxied_to,
    discrete_state_names,
    invalid_index,
):
    """Create an indexer that includes the index of proxied invalid states."""
    array_of_states_proxied_from = np.array(list_of_states_proxied_from)
    array_of_states_proxied_to = np.array(list_of_states_proxied_to)

    tuple_of_states_proxied_from = tuple(
        array_of_states_proxied_from[:, i]
        for i in range(array_of_states_proxied_from.shape[1])
    )
    tuple_of_states_proxied_to = tuple(
        array_of_states_proxied_to[:, i]
        for i in range(array_of_states_proxied_to.shape[1])
    )
    index_proxy_to = map_state_to_index[tuple_of_states_proxied_to]
    invalid_proxy_idxs = np.where(index_proxy_to == invalid_index)[0]
    if len(invalid_proxy_idxs) > 0:
        example_state_proxy_to = array_of_states_proxied_to[invalid_proxy_idxs[0]]
        invalid_state_dict_to = {
            state_name: example_state_proxy_to[i]
            for i, state_name in enumerate(discrete_state_names)
        }
        example_state_proxy_from = array_of_states_proxied_from[invalid_proxy_idxs[0]]
        invalid_state_dict_from = {
            state_name: example_state_proxy_from[i]
            for i, state_name in enumerate(discrete_state_names)
        }

        import sys

        RED = "\033[31m"  # ANSI code for red
        RESET = "\033[0m"  # ANSI code to reset color
        try:
            raise ValueError(
                f"\n\nThe state "
                f"\n\n{pd.Series(invalid_state_dict_to).to_string()}\n\n"
                f"is used as a proxy state for the state:"
                f"\n\n{pd.Series(invalid_state_dict_from).to_string()}\n\n"
                f"However, the proxy state is also declared invalid by "
                "the sparsity condition. This is not allowed. The proxy state must be valid."
            )
        except ValueError as e:
            print(f"\n\n{RED}State space error:{RESET} {e}", file=sys.stderr)
            sys.exit(1)  # Exit without showing the traceback

    map_state_to_index_with_proxies = map_state_to_index.copy()

    map_state_to_index_with_proxies[tuple_of_states_proxied_from] = index_proxy_to
    return map_state_to_index_with_proxies
