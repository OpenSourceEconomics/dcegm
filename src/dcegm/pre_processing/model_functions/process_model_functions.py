from typing import Callable, Dict, Optional

import jax.numpy as jnp

from dcegm.pre_processing.model_functions.taste_shock_function import (
    process_shock_functions,
)
from dcegm.pre_processing.model_functions.upper_evelope_wrapper import (
    create_upper_envelope_function,
)
from dcegm.pre_processing.model_structure.stochastic_states import (
    create_stochastic_transition_function,
)
from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)


def process_model_functions_and_extract_info(
    model_config: Dict,
    model_specs: Dict,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    stochastic_states_transitions: Optional[Dict[str, Callable]] = None,
    shock_functions: Optional[Dict[str, Callable]] = None,
):
    """Create wrapped functions from user supplied functions.

    Args:
        options (Dict[str, int]): Options dictionary.
        map_state_variables_to_index (Dict[str, int]): Dictionary mapping state
            variables to their index in the state vector.
        user_utility_functions (Dict[str, callable]): Dictionary of three user-supplied
            functions for computation of:
            (i) utility
            (ii) inverse marginal utility
            (iii) next period marginal utility
        user_budget_constraint (callable): Callable budget constraint.
        exogenous_transition_function (callable): User-supplied function returning for
            each state a transition matrix vector.


    Returns:
        tuple:

        - compute_utility (callable): Function for computation of agent's utility.
        - compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility. The input ```params``` is already partialled in.
        - compute_inverse_marginal_utility (Callable): Function for calculating the
            inverse marginal utility, which takes the marginal utility as only input.
        - compute_value (callable): Function for calculating the value from consumption
            level, discrete choice and expected value. The inputs ```discount_rate```
            and ```compute_utility``` are already partialled in.
        - compute_next_wealth_matrices (callable): User-defined function to compute the
            agent's wealth matrices of the next period (t + 1). The inputs
            ```savings_grid```, ```income_shocks```, ```params``` and ```options```
            are already partialled in.
        - compute_upper_envelope (Callable): Function for calculating the upper envelope
            of the policy and value function. If the number of discrete choices is 1,
            this function is a dummy function that returns the policy and value
            function as is, without performing a fast upper envelope scan.
        - transition_function (Callable): Partialled transition function that returns
            transition probabilities for each state.

    """
    # Assign continuous state name
    second_continuous_state_name = model_config["continuous_states_info"][
        "second_continuous_state_name"
    ]

    # Process mandatory functions. Start with utility functions
    compute_utility = determine_function_arguments_and_partial_model_specs(
        func=utility_functions["utility"],
        model_specs=model_specs,
        continuous_state_name=second_continuous_state_name,
    )

    compute_marginal_utility = determine_function_arguments_and_partial_model_specs(
        func=utility_functions["marginal_utility"],
        model_specs=model_specs,
        continuous_state_name=second_continuous_state_name,
    )

    compute_inverse_marginal_utility = (
        determine_function_arguments_and_partial_model_specs(
            func=utility_functions["inverse_marginal_utility"],
            model_specs=model_specs,
            continuous_state_name=second_continuous_state_name,
        )
    )

    utility_functions_processed = {
        "compute_utility": compute_utility,
        "compute_marginal_utility": compute_marginal_utility,
        "compute_inverse_marginal_utility": compute_inverse_marginal_utility,
    }
    # Final period utility functions
    compute_utility_final = determine_function_arguments_and_partial_model_specs(
        func=utility_functions_final_period["utility"],
        model_specs=model_specs,
        continuous_state_name=second_continuous_state_name,
    )

    compute_marginal_utility_final = (
        determine_function_arguments_and_partial_model_specs(
            func=utility_functions_final_period["marginal_utility"],
            model_specs=model_specs,
            continuous_state_name=second_continuous_state_name,
        )
    )

    utility_functions_final_period_processed = {
        "compute_utility_final": compute_utility_final,
        "compute_marginal_utility_final": compute_marginal_utility_final,
    }

    # Now exogenous transition function if present
    compute_stochastic_transition_vec, stochastic_transitions_dict = (
        create_stochastic_transition_function(
            stochastic_states_transitions,
            model_config=model_config,
            model_specs=model_specs,
            continuous_state_name=second_continuous_state_name,
        )
    )

    # Now state space functions
    state_specific_choice_set, next_period_deterministic_state, sparsity_condition = (
        process_state_space_functions(
            state_space_functions,
            model_config=model_config,
            model_specs=model_specs,
            continuous_state_name=second_continuous_state_name,
        )
    )

    next_period_continuous_state = process_second_continuous_update_function(
        second_continuous_state_name, state_space_functions, model_specs=model_specs
    )

    # Budget equation
    compute_assets_begin_of_period = (
        determine_function_arguments_and_partial_model_specs(
            func=budget_constraint,
            continuous_state_name=second_continuous_state_name,
            model_specs=model_specs,
        )
    )

    # Upper envelope function
    compute_upper_envelope = create_upper_envelope_function(
        model_config=model_config,
        continuous_state=second_continuous_state_name,
    )

    taste_shock_function_processed, taste_shock_scale_in_params = (
        process_shock_functions(
            shock_functions,
            model_specs,
            continuous_state_name=second_continuous_state_name,
        )
    )
    model_config_processed = model_config
    model_config_processed["params_check_info"] = {
        "taste_shock_scale_in_params": taste_shock_scale_in_params
    }

    model_funcs = {
        **utility_functions_processed,
        **utility_functions_final_period_processed,
        "compute_assets_begin_of_period": compute_assets_begin_of_period,
        "next_period_continuous_state": next_period_continuous_state,
        "sparsity_condition": sparsity_condition,
        "compute_stochastic_transition_vec": compute_stochastic_transition_vec,
        "processed_stochastic_funcs": stochastic_transitions_dict,
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_deterministic_state": next_period_deterministic_state,
        "compute_upper_envelope": compute_upper_envelope,
        "taste_shock_function": taste_shock_function_processed,
    }

    return model_funcs, model_config_processed


def process_state_space_functions(
    state_space_functions,
    model_config,
    model_specs,
    continuous_state_name,
):

    state_space_functions = (
        {} if state_space_functions is None else state_space_functions
    )

    if "state_specific_choice_set" not in state_space_functions:
        print(
            "State specific choice set not provided. Assume all choices are "
            "available in every state."
        )

        def state_specific_choice_set(**kwargs):
            return jnp.array(model_config["choices"])

    else:
        state_specific_choice_set = (
            determine_function_arguments_and_partial_model_specs(
                func=state_space_functions["state_specific_choice_set"],
                model_specs=model_specs,
                continuous_state_name=continuous_state_name,
            )
        )

    if "next_period_deterministic_state" not in state_space_functions:
        print(
            "Update function for state space not given. Assume states only change "
            "with an increase of the period and lagged choice."
        )

        def next_period_deterministic_state(**kwargs):
            return {"period": kwargs["period"] + 1, "lagged_choice": kwargs["choice"]}

    else:
        next_period_deterministic_state = (
            determine_function_arguments_and_partial_model_specs(
                func=state_space_functions["next_period_deterministic_state"],
                model_specs=model_specs,
                continuous_state_name=continuous_state_name,
            )
        )

    sparsity_condition = process_sparsity_condition(
        state_space_functions=state_space_functions, model_specs=model_specs
    )

    return (
        state_specific_choice_set,
        next_period_deterministic_state,
        sparsity_condition,
    )


def process_sparsity_condition(state_space_functions, model_specs):
    if "sparsity_condition" in state_space_functions.keys():
        sparsity_condition = determine_function_arguments_and_partial_model_specs(
            func=state_space_functions["sparsity_condition"], model_specs=model_specs
        )
        # ToDo: Error if sparsity condition takes second continuous state as input
    else:
        print("Sparsity condition not provided. Assume all states are valid.")

        def sparsity_condition(**kwargs):
            return True

    return sparsity_condition


def process_second_continuous_update_function(
    continuous_state_name, state_space_functions, model_specs
):
    if continuous_state_name is not None:
        func_name = f"next_period_{continuous_state_name}"

        next_period_continuous_state = (
            determine_function_arguments_and_partial_model_specs(
                func=state_space_functions[func_name],
                model_specs=model_specs,
                continuous_state_name=continuous_state_name,
            )
        )
    else:
        next_period_continuous_state = None

    return next_period_continuous_state
