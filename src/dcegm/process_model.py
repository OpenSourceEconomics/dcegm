import functools
import inspect
from functools import partial
from functools import reduce
from typing import Callable
from typing import Dict
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
    user_utility_functions: Dict[str, Callable],
    user_budget_constraint: Callable,
    user_final_period_solution: Callable,
    # exogenous_transition_function: Callable,
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

    if "exogenous_processes" not in options:
        exog_mapping = jnp.array([1])
        options["state_variables"]["exogenous"] = {"exog_state": [0]}
        compute_exog_transition_vec = _return_one
    else:
        exog_mapping = create_exog_mapping(options)
        exog_funcs, _signature = process_exog_funcs_new(options)

        compute_exog_transition_vec = partial(
            get_exog_transition_vec, exog_mapping=exog_mapping, exog_funcs=exog_funcs
        )

    compute_utility = _get_function_with_filtered_args_and_kwargs(
        user_utility_functions["utility"],
        options=options,
        exog_mapping=exog_mapping,
    )
    compute_marginal_utility = _get_function_with_filtered_args_and_kwargs(
        user_utility_functions["marginal_utility"],
        options=options,
        exog_mapping=exog_mapping,
    )
    compute_inverse_marginal_utility = _get_function_with_filtered_args_and_kwargs(
        user_utility_functions["inverse_marginal_utility"],
        options=options,
        exog_mapping=exog_mapping,
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
        ),
        options=options,
    )

    # ! update endgo also partial !

    if len(options["state_variables"]["choice"]) < 2:
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
    )


def _return_one(*args, **kwargs):
    return jnp.array([1])


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


def _get_vmapped_function_with_args_and_filtered_kwargs(func, options):
    signature = list(inspect.signature(func).parameters)

    @functools.wraps(func)
    def processed_func(*args):
        _args = [arg for arg in args if not isinstance(arg, dict)]
        _options_and_params = [arg for arg in args if isinstance(arg, dict)]

        # _kwargs = {
        #     key: _dict.pop(key)
        #     for key in signature
        #     for _dict in _options_and_params
        #     if key in _dict and key != "options"
        # }
        _kwargs = {}
        if "params" in signature:
            idx = 1 if len(_options_and_params) == 2 else 0
            _kwargs["params"] = _options_and_params[idx]  # fix this! flexible position

        # partial in
        if "options" in signature and "model_params" in options:
            _kwargs["options"] = options["model_params"]

        return func(*_args, **_kwargs)

    return processed_func


def _get_function_with_filtered_args_and_kwargs(func, options, exog_mapping):
    """The order of inputs in the user function does not matter!

    ;)

    """
    signature = list(inspect.signature(func).parameters)

    exog_state_vars = options["state_variables"]["exogenous"]
    exog_to_index = {key: idx for idx, key in enumerate(exog_state_vars.keys())}
    exog_keys_in_signature = signature & exog_state_vars.keys()

    endog_state_vars = options["state_variables"]["endogenous"]
    endog_to_index = {key: idx for idx, key in enumerate(endog_state_vars.keys())}

    @functools.wraps(func)
    def processed_func(*args, **kwargs):
        exog_state_global = args[-2]
        endog_args, choice = args[:-2], args[-1]

        _args_to_kwargs = {
            key: endog_args[idx]
            for key, idx in endog_to_index.items()
            if key in signature
        }

        if "choice" in signature:
            _args_to_kwargs["choice"] = choice

        if exog_keys_in_signature:
            exog_states = exog_mapping[exog_state_global]
            _args_to_kwargs.update(
                {
                    key: exog_states[idx]
                    for key, idx in exog_to_index.items()
                    if key in signature
                }
            )

        # partial in
        if "options" in signature and "model_params" in options:
            _args_to_kwargs["options"] = options["model_params"]

        _kwargs = {
            key: kwargs[key]
            for key in signature
            if key in kwargs  # and key != "options"
        }

        return func(**_args_to_kwargs | _kwargs)

    # Set name of the original func
    processed_func.__name__ = func.__name__

    return processed_func


def _get_function_with_filtered_kwargs(func, options, exog_mapping):
    """For funcs that do not take state_vec."""
    signature = list(inspect.signature(func).parameters)

    # exog state global to exog states
    signature = list(inspect.signature(func).parameters)
    (signature & options["state_variables"]["exogenous"].keys()).pop()

    _endog_state_vars_and_choice = options["state_variables"]["endogenous"] | {
        "choice": options["state_variables"]["choice"]
    }
    endog_to_index = {
        key: idx for idx, key in enumerate(_endog_state_vars_and_choice.keys())
    }

    @functools.wraps(func)
    def processed_func(*args, **kwargs):
        # _args = [arg for arg in args if not isinstance(arg, dict)]
        exog_state_global = args[-2]
        exog_mapping[exog_state_global]

        _args = [
            args[idx]
            for key, idx in endog_to_index.items()
            if key in signature and key != "choice"  # and key not in kwargs
        ]

        # exog_arg, endog_args = args[0], args[1:-1]

        # # Allows for flexible position of arguments in user function.
        # # The endog states and choice variable in the state_choice_vec
        # # (passed as args to the user function) have the same order
        # # as they appear in options["state_variables"]
        # _args_to_kwargs = {
        #     key: endog_args[idx]
        #     for key, idx in endog_to_index.items()
        #     if key in signature  # and key != "choice"
        # }

        # _args_to_kwargs[exog_key] = exog_arg

        # _choice

        _kwargs = {
            key: kwargs[key] for key in signature if key in kwargs and key != "options"
        }

        # Allow for flexible position of "choice" argument in user function
        if "choice" in signature:
            _kwargs["choice"] = args[signature.index("choice")]

        # partial in
        if "options" in signature and "model_params" in options:
            _kwargs["options"] = options["model_params"]

        return func(*_args, **_kwargs)

    # Set name of the original func
    processed_func.__name__ = func.__name__

    return processed_func


def _return_policy_and_value(
    endog_grid, policy, value, expected_value_zero_savings, *args
):
    endog_grid = jnp.append(0, endog_grid)
    policy = jnp.append(0, policy)
    value = jnp.append(expected_value_zero_savings, value)

    return endog_grid, policy, policy, value


def create_exog_transition_mat_recursively(
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
            state_var_values = [
                state_vars[i][state_indices[i]] for i in range(len(state_vars))
            ]
            exog_var_values = [
                exog_vars[i][exog_indices[i]] for i in range(len(exog_vars))
            ]

            _row = exog_indices[-1] + sum(
                i * len(var) for i, var in zip(exog_indices[:-1], exog_vars)
            )

            transition_mat[np.s_[_row, :] + tuple(state_indices)] = reduce(
                np.kron,
                [
                    func(*state_var_values, *exog_var_values, **kwargs)
                    for func in exog_funcs
                ],
            )

        else:
            for exog_i in range(len(exog_vars[0])):
                create_exog_transition_mat_recursively(
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
            create_exog_transition_mat_recursively(
                transition_mat,
                state_vars,
                exog_vars,
                state_indices + [state_i],
                exog_indices,
                exog_funcs,
                **kwargs
            )


def process_exog_funcs(options, state_vars_to_index):
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
                _get_function_with_filtered_kwargs(exog, options, state_vars_to_index),
            ]
            signature += list(inspect.signature(exog).parameters)
        # elif isinstance(exog, list):
        #     if len(exog) == 1:
        #         exog_funcs += [
        #             [
        #                 _get_opposite_prob(exog[0]),
        #                 _get_function_with_filtered_kwargs(
        #                     exog[0], options, state_vars_to_index
        #                 ),
        #             ]
        #         ]
        #         signature += list(inspect.signature(exog[0]).parameters)
        #     else:
        #         _group = []
        #         for func in exog:
        #             _group += [
        #                 _get_function_with_filtered_kwargs(
        #                     func, options, state_vars_to_index
        #                 )
        #             ]

        #             signature += list(inspect.signature(func).parameters)

        #         exog_funcs += [_group]
        # elif isinstance(exog, (np.ndarray, jnp.ndarray)):
        #     for row in exog:
        #         exog_funcs += [[_dummy_prob(prob) for prob in row]]

    return exog_funcs, list(set(signature))


# ====================================================================================


def process_exog_funcs_new(options):
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
                _get_exog_function_with_filtered_args(exog, options),
            ]
            signature += list(inspect.signature(exog).parameters)

    return exog_funcs, list(set(signature))


def _get_exog_function_with_filtered_args(func, options):
    """Args is state_vec with one global exog state."""
    signature = list(inspect.signature(func).parameters)
    exog_key = signature & options["state_variables"]["exogenous"].keys()

    _endog_state_vars_and_choice = options["state_variables"]["endogenous"] | {
        "choice": options["state_variables"]["choice"]
    }
    endog_to_index = {
        key: idx for idx, key in enumerate(_endog_state_vars_and_choice.keys())
    }

    @functools.wraps(func)
    def processed_func(*args):
        exog_arg, endog_args = args[0], args[1:-1]

        # Allows for flexible position of arguments in user function.
        # The endog states and choice variable in the state_choice_vec
        # (passed as args to the user function) have the same order
        # as they appear in options["state_variables"]
        _args_to_kwargs = {
            key: endog_args[idx]
            for key, idx in endog_to_index.items()
            if key in signature  # and key != "choice"
        }

        if exog_key:
            _args_to_kwargs[exog_key.pop()] = exog_arg

        if "params" in signature:
            _args_to_kwargs["params"] = args[-1]

        # partial in OUTSIDE!
        if "options" in signature and "model_params" in options:
            _args_to_kwargs["options"] = options["model_params"]

        return func(**_args_to_kwargs)

    # Set name of the original func
    processed_func.__name__ = func.__name__

    return processed_func


def get_exog_transition_vec(state_choice_vec, exog_mapping, exog_funcs, params):
    exog_state_global = state_choice_vec[-2]

    _idx_endog_states_and_choice = list(range(len(state_choice_vec) - 2)) + [-1]
    endog_states_and_choice = state_choice_vec[jnp.array(_idx_endog_states_and_choice)]

    trans_vecs = []

    for exog_state, exog_func in zip(exog_mapping[exog_state_global], exog_funcs):
        # options already partialled in
        trans_vec = exog_func(exog_state, *endog_states_and_choice, params)
        trans_vecs.append(trans_vec)

    trans_vec_kron = reduce(np.kron, trans_vecs)

    return jnp.array(trans_vec_kron)


def create_exog_mapping(options):
    """Create mapping from separate exog state variables to global exog state."""

    exog_state_vars = options["state_variables"]["exogenous"]

    n_elements = []
    for key in exog_state_vars:
        n_elements.append(len(exog_state_vars[key]))

    def recursive_generator(n_elements, current_mapping=[]):
        if not n_elements:
            exog_mapping.append(current_mapping)
            return

        for i in range(n_elements[0]):
            recursive_generator(n_elements[1:], current_mapping + [i])

    exog_mapping = []
    recursive_generator(n_elements)

    return jnp.asarray(exog_mapping)
