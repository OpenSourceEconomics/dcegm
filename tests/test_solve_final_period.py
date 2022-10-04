from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from dcegm.solve import solve_final_period
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model import utility_func_crra


# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


def get_example_model(model):
    """Return parameters and options of an example model."""
    params = pd.read_csv(
        TEST_RESOURCES_DIR / f"{model}.csv", index_col=["category", "name"]
    )
    options = yaml.safe_load((TEST_RESOURCES_DIR / f"{model}.yaml").read_text())
    return params, options


model = ["deaton", "retirement_taste_shocks", "retirement_no_taste_shocks"]
max_wealth = [50, 11]
n_grid_points = [1000, 101]
TEST_CASES = list(product(model, max_wealth, n_grid_points))


@pytest.mark.parametrize("model, max_wealth, n_grid_points", TEST_CASES)
def test_consume_everything_in_final_period(model, max_wealth, n_grid_points):
    params, options = get_example_model(f"{model}")
    savings_grid = np.linspace(0, max_wealth, n_grid_points)

    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]
    _periods = np.arange(n_periods)
    _choices = np.arange(n_choices)

    state_space = np.column_stack(
        [np.repeat(_periods, n_choices), np.tile(_choices, n_periods)]
    )

    condition_final_period = np.where(state_space[:, 0] == n_periods - 1)
    states_final_period = state_space[condition_final_period]
    n_states = states_final_period.shape[0]

    policy_final, value_final = solve_final_period(
        states=states_final_period,
        savings_grid=savings_grid,
        params=params,
        options=options,
        compute_utility=utility_func_crra,
    )

    _savings_grid = np.concatenate(([0], savings_grid))
    policy_final_expected = np.tile(_savings_grid, (2, 1))

    for state_index in range(n_states):
        for choice in range(n_choices):
            _utility = np.concatenate(
                ([0, 0], utility_func_crra(savings_grid[1:], choice, params))
            )
            value_final_expected = np.row_stack((_savings_grid, _utility))

            aaae(
                policy_final[state_index, choice, :][
                    :,
                    ~np.isnan(policy_final[state_index, choice, :]).any(axis=0),
                ],
                policy_final_expected,
            )
            aaae(
                value_final[state_index, choice, :][
                    :,
                    ~np.isnan(value_final[state_index, choice, :]).any(axis=0),
                ],
                value_final_expected,
            )
