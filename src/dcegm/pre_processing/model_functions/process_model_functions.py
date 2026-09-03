from typing import Callable, Dict, Optional

import jax
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
    try_jax_array,
)


def process_model_functions_and_extract_info(
    model_config: Dict,
    model_specs: Dict,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    utility_functions_final_period: Dict[str, Callable],
    budget_constraint: Callable,
    stochastic_states_transitions: Optional[Dict[str, Callable]],
    shock_functions: Optional[Dict[str, Callable]],
    continuous_grid_functions: Optional[Dict[str, Callable]],
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
    # Assign continuous-state information.
    additional_continuous_state_names = model_config["continuous_states_info"][
        "additional_continuous_state_names"
    ]
    has_additional_continuous_states = len(additional_continuous_state_names) > 0

    # We use this for functions which are called later in the jitted code
    model_specs_jax = jax.tree_util.tree_map(try_jax_array, model_specs)

    # Process mandatory functions. Start with utility functions
    compute_utility = determine_function_arguments_and_partial_model_specs(
        func=utility_functions["utility"],
        model_specs=model_specs_jax,
        not_allowed_state_choices=[],
    )

    compute_marginal_utility = determine_function_arguments_and_partial_model_specs(
        func=utility_functions["marginal_utility"],
        model_specs=model_specs_jax,
        not_allowed_state_choices=[],
    )

    compute_inverse_marginal_utility = (
        determine_function_arguments_and_partial_model_specs(
            func=utility_functions["inverse_marginal_utility"],
            model_specs=model_specs_jax,
            not_allowed_state_choices=[],
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
        model_specs=model_specs_jax,
        not_allowed_state_choices=[],
    )

    compute_marginal_utility_final = (
        determine_function_arguments_and_partial_model_specs(
            func=utility_functions_final_period["marginal_utility"],
            model_specs=model_specs_jax,
            not_allowed_state_choices=[],
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
            model_specs=model_specs_jax,
        )
    )

    # Now state space functions - here we use the old model_specs
    state_specific_choice_set, next_period_deterministic_state, sparsity_condition = (
        process_state_space_functions(
            state_space_functions,
            model_config=model_config,
            model_specs=model_specs,
            additional_continuous_state_names=additional_continuous_state_names,
        )
    )

    next_period_continuous_state = process_second_continuous_update_function(
        state_space_functions=state_space_functions,
        model_specs=model_specs_jax,
        has_additional_continuous_states=has_additional_continuous_states,
    )

    continuous_grid_functions_processed, state_specific_continuous_grid_names = (
        process_continuous_grid_functions(
            continuous_grid_functions=continuous_grid_functions,
            model_config=model_config,
            model_specs=model_specs_jax,
        )
    )

    # Budget equation
    compute_assets_begin_of_period = (
        determine_function_arguments_and_partial_model_specs(
            func=budget_constraint,
            model_specs=model_specs_jax,
            not_allowed_state_choices=[],
        )
    )

    # Upper envelope function
    compute_upper_envelope = create_upper_envelope_function(
        model_config=model_config,
        continuous_grid_functions=continuous_grid_functions_processed,
    )

    taste_shock_function_processed, taste_shock_scale_in_params = (
        process_shock_functions(
            shock_functions=shock_functions,
            model_specs=model_specs,
            model_specs_jax=model_specs_jax,
            additional_continuous_state_names=additional_continuous_state_names,
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
        "continuous_grid_functions": continuous_grid_functions_processed,
        "state_specific_continuous_grid_names": state_specific_continuous_grid_names,
    }

    return model_funcs, model_config_processed


def process_state_space_functions(
    state_space_functions,
    model_config,
    model_specs,
    additional_continuous_state_names,
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
        not_allowed_state_choices = ["assets_begin_of_period"] + list(
            additional_continuous_state_names
        )

        state_specific_choice_set = (
            determine_function_arguments_and_partial_model_specs(
                func=state_space_functions["state_specific_choice_set"],
                model_specs=model_specs,
                not_allowed_state_choices=not_allowed_state_choices,
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
                not_allowed_state_choices=[],
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
            func=state_space_functions["sparsity_condition"],
            model_specs=model_specs,
            not_allowed_state_choices=[],
        )
        # ToDo: Error if sparsity condition takes second continuous state as input
    else:
        print("Sparsity condition not provided. Assume all states are valid.")

        def sparsity_condition(**kwargs):
            return True

    return sparsity_condition


def process_second_continuous_update_function(
    state_space_functions,
    model_specs,
    has_additional_continuous_states,
):

    if has_additional_continuous_states:
        if state_space_functions is None:
            state_space_functions = {}
        if "next_period_continuous_state" not in state_space_functions:
            raise ValueError(
                "If additional continuous states are defined, provide "
                "'next_period_continuous_state' in state_space_functions."
            )
        next_period_continuous_state = (
            determine_function_arguments_and_partial_model_specs(
                func=state_space_functions["next_period_continuous_state"],
                model_specs=model_specs,
                not_allowed_state_choices=[],
            )
        )
    else:
        next_period_continuous_state = None

    return next_period_continuous_state


def process_continuous_grid_functions(
    continuous_grid_functions, model_config, model_specs
):
    """Wrap every continuous state's grid into one uniform callable.

    ``continuous_grid_functions`` is an optional user-supplied
    ``Dict[str, Callable]``, following the same convention as
    ``shock_functions``/``stochastic_states_transitions``: a top-level argument to
    ``create_model_dict``, not something nested inside ``model_config`` (which
    holds data/config, never functions). Keys are continuous-state names (any of
    the additional continuous states, ``assets_end_of_period``, or, if present,
    ``assets_begin_of_period``); values are callables
    ``(**discrete_state, choice) -> 1d array``, processed the same way as
    ``sparsity_condition``/``next_period_deterministic_state`` -- grids live on
    the state-choice space (that's where the solution itself lives, see
    ``continuous_state_grids.py``), so a grid may depend on ``choice`` too, not
    just the discrete state.

    Every continuous state of the model gets a callable in the returned dict,
    state-choice-specified or not: names left unspecified keep today's behavior
    via a callable that ignores its input and always returns the one global grid
    declared in ``model_config["continuous_states"]``.

    These are evaluated on demand wherever a batch needs a state-choice's own
    grid during solving, not precomputed for the whole state-choice space, to
    avoid materializing an ``(n_state_choices, n_points)`` table. The one
    exception is the one-time, NumPy-side consistency check in
    ``continuous_state_grids.py``, run once during model-structure construction,
    for which this function also returns the list of state-specific names (so
    that check can skip names left at their default, which are trivially
    consistent).

    """
    continuous_grid_functions = (
        {} if continuous_grid_functions is None else continuous_grid_functions
    )
    if not isinstance(continuous_grid_functions, dict):
        raise ValueError("continuous_grid_functions must be a dictionary.")

    continuous_states_info = model_config["continuous_states_info"]
    default_grids = dict(continuous_states_info["additional_continuous_state_grids"])
    default_grids["assets_end_of_period"] = continuous_states_info[
        "assets_grid_end_of_period"
    ]
    if "assets_begin_of_period" in continuous_states_info:
        default_grids["assets_begin_of_period"] = continuous_states_info[
            "assets_begin_of_period"
        ]

    for name, grid_func in continuous_grid_functions.items():
        if name not in default_grids:
            raise ValueError(
                f"continuous_grid_functions contains the key '{name}', which is "
                f"not a continuous state of this model. Valid keys are: "
                f"{sorted(default_grids)}."
            )
        if not callable(grid_func):
            raise ValueError(
                f"continuous_grid_functions['{name}'] must be a callable of the "
                "form (**discrete_state) -> 1d array."
            )
        if name != "assets_end_of_period" and default_grids[name] is not None:
            raise ValueError(
                f"continuous_grid_functions['{name}'] is given, but "
                f"model_config['continuous_states']['{name}'] is not None. A "
                f"declared array is unused once a continuous_grid_functions entry "
                f"takes over -- set model_config['continuous_states']['{name}'] to "
                "None to make that explicit. ('assets_end_of_period' is the one "
                "exception: check_model_config.py reads its length eagerly, for "
                "every upper_envelope method, before continuous_grid_functions is "
                "processed and before any state-choice exists to pin a deferred "
                "size against -- so it must always be a real array.)"
            )

    if (
        "assets_begin_of_period" in continuous_grid_functions
        and continuous_states_info["n_additional_continuous_states"] > 0
        and not model_config["upper_envelope"]["skip_endog_grid_storage"]
    ):
        raise ValueError(
            "continuous_grid_functions['assets_begin_of_period'] together with "
            "additional continuous states requires skip_endog_grid_storage "
            "(upper_envelope['method'] == 'druedahl_jorgensen' and at least two "
            "choices). With a single choice the Druedahl-Jorgensen upper envelope "
            "is skipped entirely (see check_model_config.py), so the stored "
            "endogenous grid is not the fixed assets_begin_of_period grid there, "
            "and a state-specific assets_begin_of_period has nothing to plug "
            "into. Remove either the additional continuous state(s) or the "
            "continuous_grid_functions['assets_begin_of_period'] entry, or add a "
            "second choice."
        )

    continuous_grid_functions_processed = {}
    for name, default_grid in default_grids.items():
        if name in continuous_grid_functions:
            continuous_grid_functions_processed[name] = (
                determine_function_arguments_and_partial_model_specs(
                    func=continuous_grid_functions[name],
                    model_specs=model_specs,
                    not_allowed_state_choices=[],
                )
            )
        elif default_grid is None:
            raise ValueError(
                f"model_config['continuous_states']['{name}'] is None, but no "
                f"matching continuous_grid_functions['{name}'] was given. A `None` "
                "grid means there is no default to fall back to -- it must be "
                "paired with a continuous_grid_functions entry."
            )
        else:
            continuous_grid_functions_processed[name] = _make_constant_grid_func(
                default_grid
            )

    return continuous_grid_functions_processed, list(continuous_grid_functions.keys())


def _make_constant_grid_func(default_grid):
    """Return a callable that ignores its input and always returns ``default_grid``.

    A factory, rather than defining the closure directly in the loop above, to avoid
    Python's late-binding-closure-in-a-loop pitfall: a closure defined inside a loop
    body captures the loop *variable*, not its value at that iteration, so every closure
    created across iterations would end up returning whatever ``default_grid`` was left
    holding after the *last* iteration. Binding it as this function's own parameter
    forces a fresh value per call.

    """

    def constant_grid_func(**kwargs):
        return default_grid

    return constant_grid_func
