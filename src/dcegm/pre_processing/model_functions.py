from typing import Callable, Dict

import jax.numpy as jnp
from upper_envelope.fues_jax.fues_jax import fues_jax

from dcegm.pre_processing.exog_processes import create_exog_transition_function
from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options


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
    # First check if we have a second continuous state
    has_second_continuous_state = len(options["exog_grids"]) == 2
    # Assign name
    if has_second_continuous_state:
        continuous_state_name = options["second_continuous_state_name"]
    else:
        continuous_state_name = None

    # Process mandatory functions. Start with utility functions
    compute_utility = determine_function_arguments_and_partial_options(
        func=utility_functions["utility"],
        options=options["model_params"],
        continuous_state_name=continuous_state_name,
    )
    compute_marginal_utility = determine_function_arguments_and_partial_options(
        func=utility_functions["marginal_utility"],
        options=options["model_params"],
        continuous_state_name=continuous_state_name,
    )
    compute_inverse_marginal_utility = determine_function_arguments_and_partial_options(
        func=utility_functions["inverse_marginal_utility"],
        options=options["model_params"],
        continuous_state_name=continuous_state_name,
    )
    # Final period utility functions
    compute_utility_final = determine_function_arguments_and_partial_options(
        func=utility_functions_final_period["utility"],
        options=options["model_params"],
        continuous_state_name=continuous_state_name,
    )

    compute_marginal_utility_final = determine_function_arguments_and_partial_options(
        func=utility_functions_final_period["marginal_utility"],
        options=options["model_params"],
        continuous_state_name=continuous_state_name,
    )

    # Now exogenous transition function if present
    compute_exog_transition_vec, processed_exog_funcs_dict = (
        create_exog_transition_function(
            options, continuous_state_name=continuous_state_name
        )
    )

    # Now state space functions
    get_state_specific_choice_set, get_next_period_state, update_continuous_state = (
        process_state_space_functions(
            state_space_functions, options, continuous_state_name
        )
    )

    # Budget equation
    compute_beginning_of_period_wealth = (
        determine_function_arguments_and_partial_options(
            func=budget_constraint,
            options=options["model_params"],
            continuous_state_name=continuous_state_name,
        )
    )

    # Upper envelope function
    compute_upper_envelope = create_upper_envelope_function(
        options,
        continuous_state=continuous_state_name,
    )

    model_funcs = {
        "compute_utility": compute_utility,
        "compute_marginal_utility": compute_marginal_utility,
        "compute_inverse_marginal_utility": compute_inverse_marginal_utility,
        "compute_utility_final": compute_utility_final,
        "compute_marginal_utility_final": compute_marginal_utility_final,
        "compute_beginning_of_period_wealth": compute_beginning_of_period_wealth,
        "update_continuous_state": update_continuous_state,
        "compute_exog_transition_vec": compute_exog_transition_vec,
        "processed_exog_funcs": processed_exog_funcs_dict,
        "get_state_specific_choice_set": get_state_specific_choice_set,
        "get_next_period_state": get_next_period_state,
        "compute_upper_envelope": compute_upper_envelope,
    }

    return model_funcs


def process_state_space_functions(
    state_space_functions, options, continuous_state_name
):

    state_space_functions = (
        {} if state_space_functions is None else state_space_functions
    )

    if "get_state_specific_choice_set" not in state_space_functions:
        print(
            "State specific choice set not provided. Assume all choices are "
            "available in every state."
        )

        def get_state_specific_choice_set(**kwargs):
            return jnp.array(options["state_space"]["choices"])

    else:
        get_state_specific_choice_set = (
            determine_function_arguments_and_partial_options(
                func=state_space_functions["get_state_specific_choice_set"],
                options=options["model_params"],
                continuous_state_name=continuous_state_name,
            )
        )

    if "get_next_period_state" not in state_space_functions:
        print(
            "Update function for state space not given. Assume states only change "
            "with an increase of the period and lagged choice."
        )

        def get_next_period_state(**kwargs):
            return {"period": kwargs["period"] + 1, "lagged_choice": kwargs["choice"]}

    else:
        get_next_period_state = determine_function_arguments_and_partial_options(
            func=state_space_functions["get_next_period_state"],
            options=options["model_params"],
            continuous_state_name=continuous_state_name,
        )

    if continuous_state_name is not None:
        func_name = next(
            (
                key
                for key in state_space_functions
                for name in [
                    "continuous_state",
                    continuous_state_name,
                ]
                if f"get_next_period_{name}" in key
                or f"get_next_{name}" in key
                or f"update_{name}" in key
            ),
            None,
        )

        update_continuous_state = determine_function_arguments_and_partial_options(
            func=state_space_functions[func_name],
            options=options["model_params"],
            continuous_state_name=continuous_state_name,
        )
    else:
        update_continuous_state = None

    return get_state_specific_choice_set, get_next_period_state, update_continuous_state


def create_upper_envelope_function(options, continuous_state=None):
    if len(options["state_space"]["choices"]) < 2:
        compute_upper_envelope = _return_policy_and_value
    else:

        if continuous_state:

            def compute_upper_envelope(
                endog_grid,
                policy,
                value,
                expected_value_zero_savings,
                second_continuous_state,
                state_choice_dict,
                utility_function,
                params,
            ):
                value_kwargs = {
                    "second_continuous_state": second_continuous_state,
                    "expected_value_zero_savings": expected_value_zero_savings,
                    "params": params,
                    **state_choice_dict,
                }

                def value_function(
                    consumption,
                    second_continuous_state,
                    expected_value_zero_savings,
                    params,
                    **state_choice_dict,
                ):
                    return (
                        utility_function(
                            consumption=consumption,
                            continuous_state=second_continuous_state,
                            params=params,
                            **state_choice_dict,
                        )
                        + params["beta"] * expected_value_zero_savings
                    )

                return fues_jax(
                    endog_grid=endog_grid,
                    policy=policy,
                    value=value,
                    expected_value_zero_savings=expected_value_zero_savings,
                    value_function=value_function,
                    value_function_kwargs=value_kwargs,
                    n_constrained_points_to_add=options["tuning_params"][
                        "n_constrained_points_to_add"
                    ],
                    n_final_wealth_grid=endog_grid.shape[0]
                    * (1 + options["tuning_params"]["extra_wealth_grid_factor"]),
                )

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
                value_kwargs = {
                    "expected_value_zero_savings": expected_value_zero_savings,
                    "params": params,
                    **state_choice_dict,
                }

                def value_function(
                    consumption,
                    expected_value_zero_savings,
                    params,
                    **state_choice_dict,
                ):
                    return (
                        utility_function(
                            consumption=consumption, params=params, **state_choice_dict
                        )
                        + params["beta"] * expected_value_zero_savings
                    )

                return fues_jax(
                    endog_grid=endog_grid,
                    policy=policy,
                    value=value,
                    expected_value_zero_savings=expected_value_zero_savings,
                    value_function=value_function,
                    value_function_kwargs=value_kwargs,
                    n_constrained_points_to_add=options["tuning_params"][
                        "n_constrained_points_to_add"
                    ],
                    n_final_wealth_grid=endog_grid.shape[0]
                    * (1 + options["tuning_params"]["extra_wealth_grid_factor"]),
                )

    return compute_upper_envelope


def _return_policy_and_value(
    endog_grid, policy, value, expected_value_zero_savings, *args
):
    """This is a dummy function for the case of only one discrete choice."""
    n_nans = int(0.2 * endog_grid.shape[0])

    nans_to_append = jnp.full(n_nans - 1, jnp.nan)
    endog_grid = jnp.append(jnp.append(0, endog_grid), nans_to_append)
    policy = jnp.append(jnp.append(0, policy), nans_to_append)
    value = jnp.append(jnp.append(expected_value_zero_savings, value), nans_to_append)

    return endog_grid, policy, value
