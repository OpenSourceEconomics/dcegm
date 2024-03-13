import glob
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import solve_dcegm
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.state_space_objects import update_state
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_final_consume_all,
)
from toy_models.consumption_retirement_model.utility_functions import (
    utility_final_consume_all,
)

from tests.two_period_models.model import budget_dcegm_exog_ltc
from tests.two_period_models.model import budget_dcegm_exog_ltc_and_job_offer
from tests.two_period_models.model import (
    flow_utility,
)
from tests.two_period_models.model import (
    inverse_marginal_utility,
)
from tests.two_period_models.model import (
    marginal_utility,
)
from tests.two_period_models.model import prob_exog_job_offer
from tests.two_period_models.model import prob_exog_ltc


# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"

WEALTH_GRID_POINTS = 100


# Add the utils directory to the path so that we can import helper functions.
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


def pytest_sessionstart(session):  # noqa: ARG001
    jax.config.update("jax_enable_x64", val=True)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    # Get the current working directory
    cwd = os.getcwd()

    # Search for .npy files that match the naming pattern
    pattern = os.path.join(cwd, "[endog_grid_, policy_, value_]*.npy")
    npy_files = glob.glob(pattern)

    # Delete the matching .npy files
    for file in npy_files:
        os.remove(file)


@pytest.fixture(scope="session")
def load_example_model():
    def load_options_and_params(model):
        """Return parameters and options of an example model."""
        params = pd.read_csv(
            REPLICATION_TEST_RESOURCES_DIR / f"{model}" / "params.csv",
            index_col=["category", "name"],
        )
        params = (
            params.reset_index()[["name", "value"]].set_index("name")["value"].to_dict()
        )
        options = yaml.safe_load(
            (REPLICATION_TEST_RESOURCES_DIR / f"{model}" / "options.yaml").read_text()
        )
        return params, options

    return load_options_and_params


@pytest.fixture(scope="session")
def state_space_functions():
    """Return dict with state space functions."""
    out = {
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
        "get_next_period_state": update_state,
    }
    return out


@pytest.fixture(scope="session")
def utility_functions():
    """Return dict with utility functions."""
    out = {
        "utility": flow_utility,
        "marginal_utility": marginal_utility,
        "inverse_marginal_utility": inverse_marginal_utility,
    }
    return out


@pytest.fixture(scope="session")
def utility_functions_final_period():
    """Return dict with utility functions for final period."""
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


@pytest.fixture(scope="session")
def params_and_options_exog_ltc():
    params = {}
    params["rho"] = 0.5
    params["delta"] = 0.5
    params["interest_rate"] = 0.02
    params["ltc_cost"] = 5
    params["wage_avg"] = 8
    params["sigma"] = 1
    params["lambda"] = 10
    params["beta"] = 0.95

    # exog params
    params["ltc_prob_constant"] = 0.3
    params["ltc_prob_age"] = 0.1
    params["job_offer_constant"] = 0.5
    params["job_offer_age"] = 0
    params["job_offer_educ"] = 0
    params["job_offer_type_two"] = 0.4

    options = {
        "model_params": {
            "n_grid_points": WEALTH_GRID_POINTS,
            "max_wealth": 50,
            "quadrature_points_stochastic": 5,
            "n_choices": 2,
        },
        "state_space": {
            "n_periods": 2,
            "choices": np.arange(2),
            "endogenous_states": {
                "married": [0, 1],
            },
            "exogenous_processes": {
                "ltc": {"transition": prob_exog_ltc, "states": [0, 1]},
            },
        },
    }

    return params, options


@pytest.fixture(scope="session")
def params_and_options_exog_ltc_and_job_offer():
    # ToDo: Write this as dictionary such that it has a much nicer overview
    params = {}
    params["rho"] = 0.5
    params["delta"] = 0.5
    params["interest_rate"] = 0.02
    params["ltc_cost"] = 5
    params["wage_avg"] = 8
    params["sigma"] = 1
    params["lambda"] = 1
    params["ltc_prob"] = 0.3
    params["beta"] = 0.95

    # exog params
    params["ltc_prob_constant"] = 0.3
    params["ltc_prob_age"] = 0.1
    params["job_offer_constant"] = 0.5
    params["job_offer_age"] = 0
    params["job_offer_educ"] = 0
    params["job_offer_type_two"] = 0.4

    options = {
        "model_params": {
            "n_grid_points": WEALTH_GRID_POINTS,
            "max_wealth": 50,
            "quadrature_points_stochastic": 5,
            "n_choices": 2,
        },
        "state_space": {
            "n_periods": 2,
            "choices": np.arange(2),
            "endogenous_states": {
                "married": [0, 1],
            },
            "exogenous_processes": {
                "ltc": {"transition": prob_exog_ltc, "states": [0, 1]},
                "job_offer": {"transition": prob_exog_job_offer, "states": [0, 1]},
            },
        },
    }

    return params, options


@pytest.fixture(scope="session")
def toy_model_exog_ltc(
    state_space_functions,
    utility_functions,
    utility_functions_final_period,
    params_and_options_exog_ltc,
):
    params, options = params_and_options_exog_ltc
    exog_savings_grid = jnp.linspace(
        0,
        options["model_params"]["max_wealth"],
        options["model_params"]["n_grid_points"],
    )

    out = {}
    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm_exog_ltc,
    )
    out["state_choice_space"] = model["state_choice_space"]
    out["state_space_dict"] = model["state_space_dict"]
    out["state_space_names"] = model["state_space_names"]
    out["map_state_choice_to_index"] = model["map_state_choice_to_index"]
    out["exog_state_mapping"] = model["exog_mapping"]
    out["model_funcs"] = model["model_funcs"]
    out["get_next_period_state"] = model["get_next_period_state"]

    (
        out["value"],
        out["policy_left"],
        out["policy_right"],
        out["endog_grid"],
    ) = solve_dcegm(
        params,
        options,
        exog_savings_grid=exog_savings_grid,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm_exog_ltc,
    )

    out["params"] = params
    out["options"] = options
    return out


@pytest.fixture(scope="session")
def toy_model_exog_ltc_and_job_offer(
    state_space_functions,
    utility_functions,
    utility_functions_final_period,
    params_and_options_exog_ltc_and_job_offer,
):
    params, options = params_and_options_exog_ltc_and_job_offer
    exog_savings_grid = jnp.linspace(
        0,
        options["model_params"]["max_wealth"],
        options["model_params"]["n_grid_points"],
    )

    out = {}
    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm_exog_ltc_and_job_offer,
    )
    out["state_choice_space"] = model["state_choice_space"]
    out["state_space_dict"] = model["state_space_dict"]
    out["state_space_names"] = model["state_space_names"]
    out["map_state_choice_to_index"] = model["map_state_choice_to_index"]
    out["exog_state_mapping"] = model["exog_mapping"]
    out["model_funcs"] = model["model_funcs"]
    out["get_next_period_state"] = model["get_next_period_state"]

    (
        out["value"],
        out["policy_left"],
        out["policy_right"],
        out["endog_grid"],
    ) = solve_dcegm(
        params,
        options,
        exog_savings_grid=exog_savings_grid,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm_exog_ltc,
    )

    out["params"] = params
    out["options"] = options
    return out
