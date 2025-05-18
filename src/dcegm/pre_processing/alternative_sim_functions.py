import pickle
from typing import Callable, Dict

from dcegm.pre_processing.check_options import check_model_config_and_process
from dcegm.pre_processing.model_functions.process_model_functions import (
    process_second_continuous_update_function,
    process_state_space_functions,
)
from dcegm.pre_processing.model_functions.taste_shock_function import (
    process_shock_functions,
)
from dcegm.pre_processing.model_functions.upper_evelope_wrapper import (
    create_upper_envelope_function,
)
from dcegm.pre_processing.model_structure.exogenous_processes import (
    create_exog_state_mapping,
    create_exog_transition_function,
    process_exog_model_specifications,
)
from dcegm.pre_processing.shared import (
    create_array_with_smallest_int_dtype,
    determine_function_arguments_and_partial_model_specs,
)


def generate_alternative_sim_functions(
    options: Dict,
    budget_constraint: Callable,
    state_space_functions: Dict[str, Callable] = None,
    shock_functions: Dict[str, Callable] = None,
):
    """Set up the model for dcegm.

    It consists of two steps. First it processes the user supplied functions to make
    them compatible with the interface the dcegm software expects. Second it creates
    the states and choice objects used by the dcegm software.

    Args:
        options (Dict[str, int]): Options dictionary.
        state_space_functions (Dict[str, Callable]): Dictionary of user supplied
        functions for computation of:
            (i) next period endogenous states
            (ii) next period exogenous states
            (iii) next period discrete choices
        budget_constraint (Callable): User supplied budget constraint.

    """

    options = check_model_config_and_process(options)

    model_funcs = process_alternative_sim_functions(
        options,
        state_space_functions=state_space_functions,
        budget_constraint=budget_constraint,
        shock_functions=shock_functions,
    )

    (
        exog_states_names,
        exog_state_space_raw,
    ) = process_exog_model_specifications(model_config=options["state_space"])

    exog_state_space = create_array_with_smallest_int_dtype(exog_state_space_raw)

    model_funcs["exog_state_mapping"] = create_exog_state_mapping(
        exog_state_space,
        exog_states_names,
    )

    print("Model setup complete.\n")
    return model_funcs


def process_alternative_sim_functions(
    options: Dict,
    state_space_functions: Dict[str, Callable],
    budget_constraint: Callable,
    shock_functions: Dict[str, Callable] = None,
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
    # First check if we have a second continuous state
    has_second_continuous_state = len(options["exog_grids"]) == 2
    # Assign name
    if has_second_continuous_state:
        continuous_state_name = options["second_continuous_state_name"]
    else:
        continuous_state_name = None

    # Now exogenous transition function if present
    compute_exog_transition_vec, processed_exog_funcs_dict = (
        create_exog_transition_function(
            options, continuous_state_name=continuous_state_name
        )
    )

    # Now state space functions
    state_specific_choice_set, next_period_endogenous_state, sparsity_condition = (
        process_state_space_functions(
            state_space_functions, options, continuous_state_name
        )
    )

    next_period_continuous_state = process_second_continuous_update_function(
        continuous_state_name, state_space_functions, options
    )

    # Budget equation
    compute_beginning_of_period_wealth = (
        determine_function_arguments_and_partial_model_specs(
            func=budget_constraint,
            model_specs=options["model_params"],
            continuous_state_name=continuous_state_name,
        )
    )

    # Upper envelope function
    compute_upper_envelope = create_upper_envelope_function(
        options,
        continuous_state=continuous_state_name,
    )

    taste_shock_function_processed = process_shock_functions(
        shock_functions, options, continuous_state_name
    )

    alt_model_funcs = {
        "compute_beginning_of_period_wealth": compute_beginning_of_period_wealth,
        "next_period_continuous_state": next_period_continuous_state,
        "sparsity_condition": sparsity_condition,
        "compute_exog_transition_vec": compute_exog_transition_vec,
        "processed_exog_funcs": processed_exog_funcs_dict,
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_endogenous_state": next_period_endogenous_state,
        "compute_upper_envelope": compute_upper_envelope,
        "taste_shock_function": taste_shock_function_processed,
    }

    return alt_model_funcs
