from typing import Dict

import numpy as np
import pandas as pd

from dcegm.pre_processing.model_structure.state_space import (
    process_endog_state_specifications,
    process_exog_model_specifications,
)


def inspect_state_space(
    options: Dict[str, float],
):
    """Creates a data frame of all potential states and a feasibility flag."""
    state_space_options = options["state_space"]
    model_params = options["model_params"]

    n_periods = state_space_options["n_periods"]
    n_choices = len(state_space_options["choices"])

    (
        add_endog_state_func,
        endog_states_names,
        n_endog_states,
        sparsity_func,
    ) = process_endog_state_specifications(
        state_space_options=state_space_options, model_params=model_params
    )

    (
        exog_states_names,
        exog_state_space,
    ) = process_exog_model_specifications(state_space_options=state_space_options)

    states_names_without_exog = ["period", "lagged_choice"] + endog_states_names

    state_space_wo_exog_list = []
    is_feasible_list = []

    for period in range(n_periods):
        for endog_state_id in range(n_endog_states):
            for lagged_choice in range(n_choices):
                # Select the endogenous state combination
                endog_states = add_endog_state_func(endog_state_id)

                # Create the state vector without the exogenous processes
                state_without_exog = [period, lagged_choice] + endog_states
                state_space_wo_exog_list += [state_without_exog]

                # Transform to dictionary to call sparsity function from user
                state_dict_without_exog = {
                    states_names_without_exog[i]: state_value
                    for i, state_value in enumerate(state_without_exog)
                }

                is_state_feasible = sparsity_func(**state_dict_without_exog)
                is_feasible_list += [is_state_feasible]

    n_exog_states = exog_state_space.shape[0]
    state_space_wo_exog = np.array(state_space_wo_exog_list)
    state_space_wo_exog_full = np.repeat(state_space_wo_exog, n_exog_states, axis=0)
    exog_state_space_full = np.tile(exog_state_space, (state_space_wo_exog.shape[0], 1))

    state_space = np.concatenate(
        (state_space_wo_exog_full, exog_state_space_full), axis=1
    )

    state_space_df = pd.DataFrame(
        state_space, columns=states_names_without_exog + exog_states_names
    )
    is_feasible_array = np.array(is_feasible_list, dtype=bool)

    state_space_df["is_feasible"] = np.repeat(is_feasible_array, n_exog_states, axis=0)

    return state_space_df
