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
import numpy as np
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

    # compute_exog_transition_probs = create_transition_function(options)
    compute_exog_transition_probs = exogenous_transition_function

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
        compute_exog_transition_probs,
    )


def create_transition_function(options, params):
    # needed !!!
    state_vars = (
        options["state_variables"]["endogenous"]
        | options["state_variables"]["exogenous"]
    )
    # flatten directly!
    state_vars_to_index = {key: idx for idx, key in enumerate(state_vars.keys())}
    # for idx, key in enumerate(state_vars.keys()):
    #     state_vars_to_index[key] = idx

    # read state vars from function

    exog_funcs_list, signature = process_exog_funcs(options)
    _filtered_state_vars = [
        key for key in signature if key in set(state_vars_to_index.keys())
    ]
    _filtered_endog_state_vars = [
        key
        for key in _filtered_state_vars
        if key in options["state_variables"]["endogenous"].keys()
    ]
    filtered_endog_state_vars_and_values = {
        key: val
        for key, val in options["state_variables"]["endogenous"].items()
        if key in _filtered_state_vars
    }
    exog_state_vars_and_values = options["state_variables"]["exogenous"]
    # create the filtered range thingy for recursion

    # cCreate transition matrix
    n_exog_states = np.prod([len(var) for var in exog_state_vars_and_values.keys()])
    result_shape = [n_exog_states, n_exog_states]
    result_shape.extend([len(var) for var in _filtered_endog_state_vars])
    transition_matrix = np.empty(result_shape)

    recursive_loop(
        transition_mat=transition_matrix,
        state_vars=filtered_endog_state_vars_and_values,
        exog_vars=exog_state_vars_and_values,
        state_indices=[],
        exog_indices=[],
        exog_funcs=exog_funcs_list,
        **params
    )

    # recursion

    transition_func = _matrix_to_func(
        transition_matrix=transition_matrix,
        filtered_state_vars=_filtered_endog_state_vars,
        state_vars_to_index=state_vars_to_index,
    )

    return transition_func


def _matrix_to_func(transition_matrix, filtered_endog_state_vars, state_vars_to_index):
    # state_vars_filtered = ["age", "married"]
    # signature = state_vars_filtered

    def get_transition_vec(*args):
        _filtered_args = [
            args[idx]
            for key, idx in state_vars_to_index.items()
            if key in filtered_endog_state_vars
        ]

        # function takes no kwargs

        return transition_matrix[..., tuple(_filtered_args)]

    return get_transition_vec


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


def _get_function_with_filtered_kwargs(func, options, state_vars_to_index):
    """For funcs that doe not take state_vec."""
    signature = list(inspect.signature(func).parameters)

    @functools.wraps(func)
    def processed_func(*args, **kwargs):
        _args = [arg for arg in args if not isinstance(arg, dict)]
        _kwargs = {
            key: kwargs[key] for key in signature if key in kwargs and key != "options"
        }

        # partial in
        if "options" in signature:
            _kwargs["options"] = options

        return func(*_args, **_kwargs)

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
    transition_mat,
    state_vars,
    exog_vars,
    state_indices,
    exog_indices,
    exog_funcs,
    **kwargs
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
                transition_mat[tuple([row, col] + state_indices)] = reduce(
                    jnp.multiply,
                    [
                        func(*state_var_values, *exog_var_values, **kwargs)
                        for func in funcs
                    ],
                )

        else:
            for exog_i in range(len(exog_vars[0])):
                recursive_loop(
                    transition_mat,
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
                transition_mat,
                state_vars,
                exog_vars,
                state_indices + [state_i],
                exog_indices,
                exog_funcs,
                **kwargs
            )


def process_exog_funcs(options, state_vars_to_index=None):
    """Process exogenous functions.

    Args:
        options (dict): Options dictionary.

    Returns:
        tuple: Tuple of exogenous processes.

    """
    exog_processes = options["exogenous_processes"]

    exog_funcs = []
    signature = []

    for exog in exog_processes.values():
        if isinstance(exog, Callable):
            exog_funcs += [
                [
                    _get_opposite_prob(exog),
                    _get_function_with_filtered_kwargs(
                        exog, options, state_vars_to_index
                    ),
                ]
            ]
            signature += list(inspect.signature(exog).parameters)
        elif isinstance(exog, list):
            if len(exog) == 1:
                exog_funcs += [
                    [
                        _get_opposite_prob(exog[0]),
                        _get_function_with_filtered_kwargs(
                            exog[0], options, state_vars_to_index
                        ),
                    ]
                ]
                signature += list(inspect.signature(exog[0]).parameters)
            else:
                _group = []
                for func in exog:
                    _group += [
                        _get_function_with_filtered_kwargs(
                            func, options, state_vars_to_index
                        )
                    ]

                    signature += list(inspect.signature(func).parameters)

                exog_funcs += [_group]
        elif isinstance(exog, (np.ndarray, jnp.ndarray)):
            for row in exog:
                exog_funcs += [[_dummy_prob(prob) for prob in row]]

    return exog_funcs, list(set(signature))


def _get_opposite_prob(func, options, state_vars_to_index):
    def opposite_prob(*args, **kwargs):
        return 1 - func(*args, **kwargs)

    return _get_function_with_filtered_kwargs(
        opposite_prob, options=options, state_vars_to_index=state_vars_to_index
    )


def _dummy_prob(prob, options=None, state_vars_to_index=None):
    def dummy_prob(*args, **kwargs):
        return prob

    return _get_function_with_filtered_kwargs(
        dummy_prob, options=options, state_vars_to_index=state_vars_to_index
    )
