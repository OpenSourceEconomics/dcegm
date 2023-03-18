from functools import partial
from itertools import product

import numpy as np
import pytest
from dcegm.final_period import final_period_wrapper
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra, utiility_func_log_crra
from jax import vmap
from dcegm.pre_processing import params_todict

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
    params_dict = params_todict(params)


    condition_final_period = np.where(state_space[:, 0] == n_periods - 1)
    states_final_period = state_space[condition_final_period]
    n_states = states_final_period.shape[0]
    if params_dict["theta"] == 1:
        compute_utility = partial(        utiility_func_log_crra
, params_dict=params_dict)
    else:

        compute_utility = partial(utility_func_crra, params_dict=params_dict)




    policy_final, value_final = final_period_wrapper(
        states=states_final_period,
        savings_grid=savings_grid,
        options=options,
        compute_utility=compute_utility,
        final_period_solution=solve_final_period,
    )

    policy_final_expected = np.tile(savings_grid, (2, 1))


    for state_index in range(n_states):
        for choice in range(n_choices):

            expected_value = vmap(compute_utility, in_axes=(0, None))(
                savings_grid, choice
            )
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
                expected_value,
            )
