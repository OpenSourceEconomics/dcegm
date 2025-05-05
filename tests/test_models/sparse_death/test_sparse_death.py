import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from dcegm.pre_processing.setup_model import process_debug_string, setup_model
from dcegm.solve import get_solve_func_for_model
from tests.test_models.sparse_death.budget import budget_constraint_exp
from tests.test_models.sparse_death.exog_processes import job_offer, prob_survival
from tests.test_models.sparse_death.state_space import create_state_space_functions
from tests.test_models.sparse_death.utility import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)


@pytest.fixture
def inputs():
    n_periods = 20
    n_choices = 3

    state_space_options = {
        "min_period_batch_segments": [5, 12],
        "n_periods": n_periods,
        "choices": np.arange(n_choices, dtype=int),
        "endogenous_states": {
            "already_retired": np.arange(2, dtype=int),
        },
        "continuous_states": {
            "wealth": jnp.arange(0, 100, 5, dtype=float),
            "experience": jnp.linspace(0, 1, 7, dtype=float),
        },
        "exogenous_processes": {
            "job_offer": {
                "transition": job_offer,
                "states": [0, 1],
            },
            "survival": {
                "transition": prob_survival,
                "states": [0, 1],
            },
        },
    }

    options = {
        "state_space": state_space_options,
        "model_params": {
            "n_quad_points_stochastic": 5,
            "n_periods": n_periods,
            "n_choices": 3,
            "min_ret_period": 5,
            "max_ret_period": 10,
            "fresh_bonus": 0.1,
            "exp_scale": 20,
        },
    }

    params = {
        "delta": 0.5,
        "beta": 0.95,
        "taste_shock_scale": 1,
        "sigma": 1,
        "interest_rate": 0.05,
        "constant": 1,
        "exp": 0.1,
        "exp_squared": -0.01,
        "consumption_floor": 0.5,
    }

    model = setup_model(
        options=options,
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        state_space_functions=create_state_space_functions(),
        budget_constraint=budget_constraint_exp,
        debug_info="all",
    )
    solve_func = get_solve_func_for_model(model)

    solution = solve_func(params=params)

    return {
        "model": model,
        "params": params,
        "options": options,
        "solution": solution,
    }


def test_child_states(inputs):
    model_structure = inputs["model"]["model_structure"]
    state_names = model_structure["discrete_states_names"]
    state_choice_names = state_names + ["choice"]

    state_choice_test = {
        "period": 8,
        "lagged_choice": 2,
        "job_offer": 1,
        "survival": 1,
        "already_retired": 0,
        "choice": 2,
    }

    state_choice_tuple = tuple(
        (state_choice_test[name],) for name in state_choice_names
    )

    state_choice_idx = model_structure["map_state_choice_to_index_with_proxy"][
        state_choice_tuple
    ]
    child_states_idxs = model_structure["map_state_choice_to_child_states"][
        state_choice_idx
    ]
    child_states_idxs = np.squeeze(child_states_idxs)

    child_states = model_structure["state_space"][child_states_idxs]
    # Transform array into list of states by looping over rows of matrix
    child_states_list = [list(child_state) for child_state in child_states]

    # There are three possible child states. Two of them are alive states
    # and one is a dead state.
    alive_state = {
        "period": 9,
        "lagged_choice": 2,
        "survival": 1,
        "already_retired": 0,
    }
    # Create container with all child states and loop over job offer to
    # create the two alive states
    alive_states_to_check = []
    for jo in range(2):
        alive_state["job_offer"] = jo
        alive_state_list = [alive_state[name] for name in state_names]
        alive_states_to_check += [alive_state_list]

    # Check that alive states are in child states once
    for alive_state in alive_states_to_check:
        assert alive_state in child_states_list
        child_states_list.remove(alive_state)

    # Add the dead state
    dead_state = {
        "period": 19,
        "lagged_choice": 2,
        "survival": 0,
        "already_retired": 0,
        "job_offer": 0,
    }
    dead_state_list = [dead_state[name] for name in state_names]

    # Check that death state is twice in child states
    for _ in range(2):
        assert dead_state_list in child_states_list
        child_states_list.remove(dead_state_list)

    assert len(child_states_list) == 0


def test_sparse_debugging_output(inputs):
    options = inputs["options"]

    state_space_functions = create_state_space_functions()
    # state_space_functions=options["state_space"], sparsity_condition, debugging=True

    debug_dict = process_debug_string(
        debug_output="state_space_df",
        state_space_functions=state_space_functions,
        options=options,
    )
    assert isinstance(debug_dict["debug_output"], pd.DataFrame)
