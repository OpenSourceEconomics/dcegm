from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dcegm.solve import solve_dcegm
from jax.config import config
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.exogenous_processes import (
    get_transition_matrix_by_state,
)
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period_scalar,
)
from toy_models.consumption_retirement_model.simulate_stacked import simulate_stacked
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_choice_set,
)
from toy_models.consumption_retirement_model.utility_functions import (
    inverse_marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

config.update("jax_enable_x64", True)

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


@pytest.fixture()
def utility_functions():
    """Return dict with utility functions."""
    return {
        "utility": utility_func_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
        "marginal_utility": marginal_utility_crra,
    }


@pytest.fixture()
def state_space_functions():
    """Return dict with utility functions."""
    return {
        "create_state_space": create_state_space,
        "get_state_specific_choice_set": get_state_specific_choice_set,
    }


def test_simulate(utility_functions, state_space_functions, load_example_model):
    model = "retirement_no_taste_shocks"

    params, options = load_example_model(f"{model}")
    params.loc[("shocks", "lambda"), "value"] = 2.2204e-16
    params.loc[("assets", "initial_wealth_low"), "value"] = 10
    params.loc[("assets", "initial_wealth_high"), "value"] = 30

    options["n_exog_processes"] = 1

    state_space, indexer = create_state_space(options)

    endog_grid, policy, value = solve_dcegm(
        params,
        options,
        utility_functions,
        budget_constraint=budget_constraint,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
        transition_function=get_transition_matrix_by_state,
    )

    policy_stacked = np.stack([endog_grid[::2], policy[::2]], axis=2)
    value_stacked = np.stack([endog_grid[::2], value[::2]], axis=2)

    df = simulate_stacked(
        # endog_grid=endog_grid,
        policy=policy_stacked,
        value=value_stacked,
        num_periods=options["n_periods"],
        # cost_work,
        # theta,
        # beta,
        lambda_=params.loc[("shocks", "lambda"), "value"],
        sigma=params.loc[("shocks", "sigma"), "value"],
        r=params.loc[("assets", "interest_rate"), "value"],
        coeffs_age_poly=params.loc[("wage"), "value"],
        options=options,
        params=params,
        state_space=state_space,
        indexer=indexer,
        init=[
            params.loc[("assets", "initial_wealth_low"), "value"],
            params.loc[("assets", "initial_wealth_high"), "value"],
        ],
        num_sims=100,
        seed=7134,
    )

    expected = pd.read_csv(TEST_RESOURCES_DIR / "df_simulate_fedor.csv")
    aaae(df, expected)
