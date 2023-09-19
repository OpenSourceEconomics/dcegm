import functools
import inspect
from functools import partial
from functools import reduce
from itertools import product
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import jax.numpy as jnp
import pandas as pd
from dcegm.fast_upper_envelope import fast_upper_envelope_wrapper
from pybaum import get_registry
from pybaum import leaf_names
from pybaum import tree_flatten


def process_model_functions(
    options: Dict[str, float],
    state_vars: List[str],
    user_utility_functions: Dict[str, Callable],
    user_budget_constraint: Callable,
    user_final_period_solution: Callable,
    exogenous_transition_function: Callable,
) -> Tuple[Callable, Callable, Callable, Callable, Callable, Callable, Callable]:
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

    _state_vars_to_index = {key: idx for idx, key in enumerate(state_vars)}

    # compute_utility = _get_function_with_filtered_args_and_kwargs(
    #     partial(user_utility_functions["utility"], options=options),
    #     _state_vars_to_index,
    # )

    compute_utility = _get_function_with_filtered_args_and_kwargs(
        user_utility_functions["utility"],
        options=options,
        state_vars_to_index=_state_vars_to_index,
    )
    compute_marginal_utility = _get_function_with_filtered_args_and_kwargs(
        user_utility_functions["marginal_utility"],
        options=options,
        state_vars_to_index=_state_vars_to_index,
    )
    compute_inverse_marginal_utility = _get_function_with_filtered_args_and_kwargs(
        user_utility_functions["inverse_marginal_utility"],
        options=options,
        state_vars_to_index=_state_vars_to_index,
    )

    compute_beginning_of_period_wealth = (
        _get_vmapped_function_with_args_and_filtered_kwargs(
            user_budget_constraint, options=options
        )
    )
    compute_final_period = _get_vmapped_function_with_args_and_filtered_kwargs(
        partial(
            user_final_period_solution,
            compute_utility=compute_utility,
            compute_marginal_utility=compute_marginal_utility,
            # options=options,
        ),
        options=options,
    )

    # update endgo also partial

    compute_transitions_exog_states = exogenous_transition_function

    if options["n_discrete_choices"] == 1:
        compute_upper_envelope = _return_policy_and_value
    else:
        compute_upper_envelope = fast_upper_envelope_wrapper

    return (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_beginning_of_period_wealth,
        compute_final_period,
        compute_upper_envelope,
        compute_transitions_exog_states,
    )


def convert_params_to_dict(
    params: Union[dict, pd.Series, pd.DataFrame]
) -> Dict[str, float]:
    """Transforms params DataFrame into a dictionary.

    Checks if given params DataFrame contains taste shock scale, interest rate
    and discount factor.

    Args:
        params (dict or tuple or pandas.Series or pandas.DataFrame): Model parameters
            Support tuple and list as well?

    Returns:
        dict: Dictionary of model parameters.

    """
    _registry = get_registry(
        types=[
            "pandas.Series",
            "pandas.DataFrame",
        ],
        include_defaults=True,
    )

    _params, _treedef = tree_flatten(params, registry=_registry)

    values = [i for i in _params if isinstance(i, (int, float))]

    # {level: df.xs(level).to_dict('index') for level in df.index.levels[0]}

    if isinstance(params, (pd.Series, pd.DataFrame)):
        keys = _treedef.index.get_level_values(_treedef.index.names[-1]).tolist()
    else:
        keys = leaf_names(_treedef, registry=_registry)

    params_dict = dict(zip(keys, values))

    if "interest_rate" not in params_dict:
        params_dict["interest_rate"] = 0
    if "lambda" not in params_dict:
        params_dict["lambda"] = 0
    if "sigma" not in params_dict:
        params_dict["sigma"] = 0
    if "beta" not in params_dict:
        raise ValueError("Beta must be provided in params.")

    return params_dict


def _get_function_with_filtered_args_and_kwargs(func, options, state_vars_to_index):
    signature = list(inspect.signature(func).parameters)

    @functools.wraps(func)
    def processed_func(*args, **kwargs):
        _args_to_kwargs = {
            key: args[idx]
            for key, idx in state_vars_to_index.items()
            if key in signature  # and key not in kwargs
        }

        _kwargs = {
            key: kwargs[key] for key in signature if key in kwargs and key != "options"
        }

        # partial in
        if "options" in signature:
            _kwargs["options"] = options

        return func(**_args_to_kwargs | _kwargs)

    # Set the __name__ attribute of processed_func to the name of the original func
    processed_func.__name__ = func.__name__

    # if "options" in signature:
    #     processed_func = partial(processed_func, options=options)

    return processed_func


def _get_vmapped_function_with_args_and_filtered_kwargs(func, options):
    signature = list(inspect.signature(func).parameters)

    @functools.wraps(func)
    def processed_func(*args):
        _args = [arg for arg in args if not isinstance(arg, dict)]
        _options_and_params = [arg for arg in args if isinstance(arg, dict)]

        _kwargs = {
            key: _dict.pop(key)
            for key in signature
            for _dict in _options_and_params
            if key in _dict and key != "options"
        }

        # partial in
        if "options" in signature:
            _kwargs["options"] = options

        return func(*_args, **_kwargs)

    # Set the __name__ attribute of processed_func to the name of the original func
    # processed_func.__name__ = func.__name__

    return processed_func


def _get_function_with_filtered_kwargs(func):
    signature = list(inspect.signature(func).parameters)

    @functools.wraps(func)
    def processed_func(**kwargs):
        _kwargs = {key: kwargs[key] for key in signature if key in kwargs}
        return func(**_kwargs)

    processed_func.__name__ = func.__name__

    return processed_func


def _return_policy_and_value(
    endog_grid, policy, value, expected_value_zero_savings, *args
):
    endog_grid = jnp.append(0, endog_grid)
    policy = jnp.append(0, policy)
    value = jnp.append(expected_value_zero_savings, value)
    return endog_grid, policy, policy, value


def recursive_loop(
    result, state_vars, exog_vars, state_indices, exog_indices, exog_funcs, **kwargs
):
    if len(state_indices) == len(state_vars):
        if len(exog_indices) == len(exog_vars):
            # Get the values of state variables at the current indices
            state_var_values = [
                state_vars[i][state_indices[i]] for i in range(len(state_vars))
            ]

            # Get the values of exogenous variables at the current indices
            exog_var_values = [
                exog_vars[i][exog_indices[i]] for i in range(len(exog_vars))
            ]

            # row = exog_indices[-1] + sum(
            #    exog_indices[i] * len(exog_vars[i + 1])
            #    for i in range(-len(exog_vars), -1)
            # )
            row = exog_indices[-1] + sum(
                i * len(var) for i, var in zip(exog_indices[:-1], exog_vars)
            )

            for col, funcs in enumerate(product(*exog_funcs)):
                result[tuple([row, col] + state_indices)] = reduce(
                    jnp.multiply,
                    [
                        func(*state_var_values, *exog_var_values, **kwargs)
                        for func in funcs
                    ],
                )

        else:
            for exog_i in range(len(exog_vars[0])):
                recursive_loop(
                    result,
                    state_vars,
                    exog_vars,
                    state_indices,
                    exog_indices + [exog_i],
                    exog_funcs,
                    **kwargs
                )

    else:
        for state_i in range(len(state_vars[len(state_indices)])):
            recursive_loop(
                result,
                state_vars,
                exog_vars,
                state_indices + [state_i],
                exog_indices,
                exog_funcs,
                **kwargs
            )
