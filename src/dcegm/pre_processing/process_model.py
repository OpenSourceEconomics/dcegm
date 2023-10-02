from functools import partial
from typing import Callable
from typing import Dict
from typing import Union

import jax.numpy as jnp
import pandas as pd
from dcegm.pre_processing.exog_processes import get_exog_transition_vec
from dcegm.pre_processing.exog_processes import process_exog_funcs
from dcegm.pre_processing.exog_processes import return_dummy_exog_transition
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.pre_processing.state_space import process_endog_state_specifications
from dcegm.pre_processing.state_space import process_exog_model_specifications
from dcegm.pre_processing.utils import determine_function_arguments_and_partial_options
from dcegm.upper_envelope.fast_upper_envelope import fast_upper_envelope_wrapper
from pybaum import get_registry
from pybaum import tree_flatten


def process_model_functions_and_create_state_space_objects(
    options: Dict[str, float],
    user_utility_functions: Dict[str, Callable],
    user_budget_constraint: Callable,
    user_final_period_solution: Callable,
    state_space_functions: Dict[str, Callable],
):
    (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_beginning_of_period_wealth,
        compute_final_period,
        compute_exog_transition_vec,
        compute_upper_envelope,
        get_state_specific_choice_set,
        update_endog_state_by_state_and_choice,
    ) = process_model_functions(
        options,
        user_utility_functions=user_utility_functions,
        user_budget_constraint=user_budget_constraint,
        user_final_period_solution=user_final_period_solution,
        state_space_functions=state_space_functions,
    )

    (
        period_specific_state_objects,
        state_space,
    ) = create_state_space_and_choice_objects(
        options=options,
        get_state_specific_choice_set=get_state_specific_choice_set,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )
    return (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_beginning_of_period_wealth,
        compute_final_period,
        compute_exog_transition_vec,
        compute_upper_envelope,
        period_specific_state_objects,
        state_space,
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
        num_states_of_all_endog_states,
        num_endog_states,
        sparsity_func,
    ) = process_endog_state_specifications(
        state_space_options=state_space_options, model_params=model_params
    )

    (
        add_exog_state_func,
        exog_states_names,
        num_states_of_all_exog_states,
        n_exog_states,
        _,
    ) = process_exog_model_specifications(state_space_options=state_space_options)

    states_names_without_exog = ["period", "lagged_choice"] + endog_states_names

    state_space_list = []

    idx = 0
    for period in range(n_periods):
        for lagged_choice in range(n_choices):
            for endog_state_id in range(num_endog_states):
                # Select the endogenous state combination
                endog_states = add_endog_state_func(endog_state_id)

                # Create the state vector without the exogenous processes
                state_without_exog = [period, lagged_choice] + endog_states

                # Transform to dictionary to call sparsity function from user
                state_dict_without_exog = {
                    states_names_without_exog[i]: state_value
                    for i, state_value in enumerate(state_without_exog)
                }

                is_state_valid = sparsity_func(**state_dict_without_exog)
                for exog_state_id in range(n_exog_states):
                    exog_states = add_exog_state_func(exog_state_id)
                    state = state_without_exog + exog_states

                    state_space_list += [state + [is_state_valid]]
                    idx += 1

    state_space_df = pd.DataFrame(
        state_space_list,
        columns=states_names_without_exog + exog_states_names + ["is_feasible"],
    )

    return state_space_df


def process_model_functions(
    options: Dict[str, float],
    user_utility_functions: Dict[str, Callable],
    user_budget_constraint: Callable,
    user_final_period_solution: Callable,
    state_space_functions: Dict[str, Callable],
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

    if "exogenous_processes" not in options["state_space"]:
        options["state_space"]["exogenous_states"] = {"exog_state": [0]}
        compute_exog_transition_vec = return_dummy_exog_transition
    else:
        exog_funcs = process_exog_funcs(options)

        compute_exog_transition_vec = partial(
            get_exog_transition_vec, exog_funcs=exog_funcs
        )

    model_params_options = options["model_params"]
    compute_utility = determine_function_arguments_and_partial_options(
        func=user_utility_functions["utility"], options=model_params_options
    )
    compute_marginal_utility = determine_function_arguments_and_partial_options(
        func=user_utility_functions["marginal_utility"], options=model_params_options
    )
    compute_inverse_marginal_utility = determine_function_arguments_and_partial_options(
        func=user_utility_functions["inverse_marginal_utility"],
        options=model_params_options,
    )

    compute_beginning_of_period_wealth = (
        determine_function_arguments_and_partial_options(
            func=user_budget_constraint, options=model_params_options
        )
    )

    compute_final_period = determine_function_arguments_and_partial_options(
        func=user_final_period_solution,
        options=model_params_options,
        additional_partial={
            "compute_utility": compute_utility,
            "compute_marginal_utility": compute_marginal_utility,
        },
    )

    get_state_specific_choice_set = determine_function_arguments_and_partial_options(
        func=state_space_functions["get_state_specific_choice_set"],
        options=model_params_options,
    )

    update_endog_state_by_state_and_choice = (
        determine_function_arguments_and_partial_options(
            func=state_space_functions["update_endog_state_by_state_and_choice"],
            options=model_params_options,
        )
    )

    # ! update endog also partial !

    if len(options["state_space"]["choices"]) < 2:
        compute_upper_envelope = _return_policy_and_value
    else:
        compute_upper_envelope = fast_upper_envelope_wrapper

    return (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_beginning_of_period_wealth,
        compute_final_period,
        compute_exog_transition_vec,
        compute_upper_envelope,
        get_state_specific_choice_set,
        update_endog_state_by_state_and_choice,
    )


def process_params(params: Union[dict, pd.Series, pd.DataFrame]) -> Dict[str, float]:
    """Transforms params DataFrame into a dictionary.

    Checks if given params contains beta, taste shock scale, interest rate
    and discount factor.

    Args:
        params (dict or tuple or pandas.Series or pandas.DataFrame): Model parameters
            Support tuple and list as well?

    Returns:
        dict: Dictionary of model parameters.

    """

    if isinstance(params, (pd.Series, pd.DataFrame)):
        params = _convert_params_to_dict(params)

    if "interest_rate" not in params:
        params["interest_rate"] = 0
    if "lambda" not in params:
        params["lambda"] = 0
    if "sigma" not in params:
        params["sigma"] = 0
    if "beta" not in params:
        raise ValueError("beta must be provided in params.")

    return params


def _convert_params_to_dict(params: Union[pd.Series, pd.DataFrame]):
    """Converts params to dictionary."""
    _registry = get_registry(
        types=[
            "dict",
            "pandas.Series",
            "pandas.DataFrame",
        ],
        include_defaults=False,
    )
    # {level: df.xs(level).to_dict('index') for level in df.index.levels[0]}
    _params, _treedef = tree_flatten(params, registry=_registry)
    values = [i for i in _params if isinstance(i, (int, float))]
    keys = _treedef.index.get_level_values(_treedef.index.names[-1]).tolist()
    params_dict = dict(zip(keys, values))

    return params_dict


def _return_policy_and_value(
    endog_grid, policy, value, expected_value_zero_savings, *args
):
    """This is a dummy function for the case of only one discrete choice."""
    endog_grid = jnp.append(0, endog_grid)
    policy = jnp.append(0, policy)
    value = jnp.append(expected_value_zero_savings, value)

    return endog_grid, policy, policy, value
