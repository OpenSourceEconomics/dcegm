import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dcegm.pre_processing.setup_model import setup_model
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.solve import get_solve_func_for_model
from toy_models.load_example_model import load_example_models


def marriage_transition(married, options):
    trans_mat = options["marriage_trans_mat"]
    return trans_mat[married, :]


@pytest.fixture
def state_space_options():
    state_space_options = {
        "n_periods": 5,
        "choices": np.arange(2),
        "endogenous_states": {
            "experience": np.arange(2, dtype=int),
        },
        "continuous_states": {
            "wealth": np.arange(0, 100, 5, dtype=float),
        },
        "exogenous_processes": {
            "job_offer": {
                "transition": prob_exog_health_mother,
                "states": [0, 1, 2],
            },
            "health_father": {
                "transition": prob_exog_health_father,
                "states": [0, 1, 2],
            },
            "health_child": {
                "transition": prob_exog_health_child,
                "states": [0, 1],
            },
            "health_grandma": {
                "transition": prob_exog_health_grandma,
                "states": [0, 1],
            },
        },
    }

    return state_space_options
