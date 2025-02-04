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
from tests.two_period_models.model import (
    budget_dcegm_exog_ltc,
    budget_dcegm_exog_ltc_and_job_offer,
    prob_exog_job_offer,
    prob_exog_ltc,
)
from toy_models.cons_ret_model_dcegm_paper.state_space_objects import (
    create_state_space_function_dict,
)
from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)

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
def load_replication_params_and_specs():
    def load_options_and_params(model_name):
        """Return parameters and options of an example model."""
        params = yaml.safe_load(
            (
                REPLICATION_TEST_RESOURCES_DIR / f"{model_name}" / "params.yaml"
            ).read_text()
        )
        model_specs = yaml.safe_load(
            (
                REPLICATION_TEST_RESOURCES_DIR / f"{model_name}" / "options.yaml"
            ).read_text()
        )
        return params, model_specs

    return load_options_and_params


@pytest.fixture(scope="session")
def params_and_options_exog_ltc():

    params = {
        "rho": 0.5,
        "delta": 0.5,
        "interest_rate": 0.02,
        "ltc_cost": 5,
        "wage_avg": 8,
        "sigma": 1,
        "lambda": 10,
        "beta": 0.95,
        # Exogenous parameters
        "ltc_prob_constant": 0.3,
        "ltc_prob_age": 0.1,
        "job_offer_constant": 0.5,
        "job_offer_age": 0,
        "job_offer_educ": 0,
        "job_offer_type_two": 0.4,
    }

    options = {
        "model_params": {
            "quadrature_points_stochastic": 5,
            "n_choices": 2,
        },
        "state_space": {
            "n_periods": 2,
            "choices": np.arange(2),
            "endogenous_states": {
                "married": [0, 1],
            },
            "continuous_states": {
                "wealth": np.linspace(0, 50, WEALTH_GRID_POINTS),
            },
            "exogenous_processes": {
                "ltc": {"transition": prob_exog_ltc, "states": [0, 1]},
            },
        },
    }

    return params, options


@pytest.fixture(scope="session")
def params_and_options_exog_ltc_and_job_offer():

    params = {
        "rho": 0.5,
        "delta": 0.5,
        "interest_rate": 0.02,
        "ltc_cost": 5,
        "wage_avg": 8,
        "sigma": 1,
        "lambda": 1,
        "ltc_prob": 0.3,
        "beta": 0.95,
        # Exogenous parameters
        "ltc_prob_constant": 0.3,
        "ltc_prob_age": 0.1,
        "job_offer_constant": 0.5,
        "job_offer_age": 0,
        "job_offer_educ": 0,
        "job_offer_type_two": 0.4,
    }

    options = {
        "model_params": {
            "quadrature_points_stochastic": 5,
            "n_choices": 2,
        },
        "state_space": {
            "n_periods": 2,
            "choices": np.arange(2),
            "endogenous_states": {
                "married": [0, 1],
            },
            "continuous_states": {
                "wealth": np.linspace(0, 50, WEALTH_GRID_POINTS),
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
    params_and_options_exog_ltc,
):
    params, options = params_and_options_exog_ltc

    out = {}
    model = setup_model(
        options=options,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_dcegm_exog_ltc,
    )

    (
        out["value"],
        out["policy"],
        out["endog_grid"],
    ) = solve_dcegm(
        params,
        options,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_dcegm_exog_ltc,
    )

    out["model"] = model
    out["params"] = params
    out["options"] = options
    return out


@pytest.fixture(scope="session")
def toy_model_exog_ltc_and_job_offer(
    params_and_options_exog_ltc_and_job_offer,
):
    params, options = params_and_options_exog_ltc_and_job_offer

    out = {}
    model = setup_model(
        options=options,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_dcegm_exog_ltc_and_job_offer,
    )

    (
        out["value"],
        out["policy"],
        out["endog_grid"],
    ) = solve_dcegm(
        params,
        options,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_dcegm_exog_ltc,
    )

    out["model"] = model
    out["params"] = params
    out["options"] = options

    return out
