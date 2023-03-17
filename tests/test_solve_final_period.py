from functools import partial
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.final_period import solve_final_period
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

model = ["deaton", "retirement_taste_shocks", "retirement_no_taste_shocks"]
max_wealth = [50, 11]
n_grid_points = [1000, 101]
TEST_CASES = list(product(model, max_wealth, n_grid_points))


@pytest.mark.parametrize("model, max_wealth, n_grid_points", TEST_CASES)
def test_consume_everything_in_final_period(
    model, max_wealth, n_grid_points, load_example_model
):
    params, options = load_example_model(f"{model}")
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
    compute_utility = partial(utility_func_crra, params=params)

    policy_final, value_final = solve_final_period(
        states=states_final_period,
        savings_grid=savings_grid,
        options=options,
        compute_utility=compute_utility,
    )

    _savings_grid = np.concatenate(([0], savings_grid))
    policy_final_expected = np.tile(_savings_grid, (2, 1))
    value_final_expected = np.row_stack(
        (np.ones_like(_savings_grid) * np.inf, np.zeros_like(_savings_grid))
    )

    for state_index in range(n_states):
        for choice in range(n_choices):
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
