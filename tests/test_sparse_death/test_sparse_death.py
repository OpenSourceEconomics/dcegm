import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from dcegm.interface import get_n_state_choice_period
from dcegm.pre_processing.setup_model import setup_model
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.solve import get_solve_func_for_model
from tests.test_sparse_death.budget import budget_constraint_exp
from tests.test_sparse_death.exog_processes import job_offer, prob_survival
from tests.test_sparse_death.state_space import create_state_space_functions
from tests.test_sparse_death.utility import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)


@pytest.fixture
def test_inputs():
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
            "quadrature_points_stochastic": 5,
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
        "lambda": 1,
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
    )
    solve_func = get_solve_func_for_model(model)

    solution = solve_func(params=params)

    breakpoint()

    return model


def test_1(test_inputs):
    breakpoint()
