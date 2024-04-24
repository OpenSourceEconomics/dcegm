from typing import Callable
from typing import Dict

import jax.numpy as jnp
from dcegm.pre_processing.exog_processes import create_exog_transition_function
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options
from upper_envelope.fues_jax.fues_jax import fast_upper_envelope_wrapper


def process_model_functions(
    options: Dict,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
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

    compute_exog_transition_vec = create_exog_transition_function(options)

    model_params_options = options["model_params"]

    compute_utility = determine_function_arguments_and_partial_options(
        func=utility_functions["utility"], options=model_params_options
    )
    compute_marginal_utility = determine_function_arguments_and_partial_options(
        func=utility_functions["marginal_utility"], options=model_params_options
    )
    compute_inverse_marginal_utility = determine_function_arguments_and_partial_options(
        func=utility_functions["inverse_marginal_utility"],
        options=model_params_options,
    )

    compute_utility_final = determine_function_arguments_and_partial_options(
        func=utility_functions_final_period["utility"],
        options=model_params_options,
    )
    compute_marginal_utility_final = determine_function_arguments_and_partial_options(
        func=utility_functions_final_period["marginal_utility"],
        options=model_params_options,
    )

    compute_beginning_of_period_resources = (
        determine_function_arguments_and_partial_options(
            func=budget_constraint, options=model_params_options
        )
    )

    get_state_specific_choice_set = determine_function_arguments_and_partial_options(
        func=state_space_functions["get_state_specific_choice_set"],
        options=model_params_options,
    )

    get_next_period_state = determine_function_arguments_and_partial_options(
        func=state_space_functions["get_next_period_state"],
        options=model_params_options,
    )

    compute_upper_envelope = create_upper_envelope_function(options)

    model_funcs = {
        "compute_utility": compute_utility,
        "compute_marginal_utility": compute_marginal_utility,
        "compute_inverse_marginal_utility": compute_inverse_marginal_utility,
        "compute_utility_final": compute_utility_final,
        "compute_marginal_utility_final": compute_marginal_utility_final,
        "compute_beginning_of_period_resources": compute_beginning_of_period_resources,
        "compute_exog_transition_vec": compute_exog_transition_vec,
        "get_state_specific_choice_set": get_state_specific_choice_set,
        "get_next_period_state": get_next_period_state,
        "compute_upper_envelope": compute_upper_envelope,
    }

    return model_funcs


def create_upper_envelope_function(options):
    if len(options["state_space"]["choices"]) < 2:
        compute_upper_envelope = _return_policy_and_value
    else:

        def compute_upper_envelope(
            endog_grid,
            policy,
            value,
            expected_value_zero_savings,
            state_choice_dict,
            utility_function,
            params,
        ):
            utility_kwargs = {
                **state_choice_dict,
                "params": params,
            }
            return fast_upper_envelope_wrapper(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                expected_value_zero_savings=expected_value_zero_savings,
                utility_function=utility_function,
                utility_kwargs=utility_kwargs,
                disc_factor=params["beta"],
            )

    return compute_upper_envelope


def _return_policy_and_value(
    endog_grid, policy, value, expected_value_zero_savings, *args
):
    """This is a dummy function for the case of only one discrete choice."""
    nans_to_append = jnp.full(int(0.2 * endog_grid.shape[0]) - 1, jnp.nan)
    endog_grid = jnp.append(jnp.append(0, endog_grid), nans_to_append)
    policy = jnp.append(jnp.append(0, policy), nans_to_append)
    value = jnp.append(jnp.append(expected_value_zero_savings, value), nans_to_append)

    return endog_grid, policy, value
