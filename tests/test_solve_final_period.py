from functools import partial
from itertools import product

import numpy as np
import pytest
from dcegm.final_period import final_period_wrapper
from dcegm.pre_processing import get_partial_functions
from dcegm.pre_processing import params_todict
from jax import vmap
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.exogenous_processes import (
    get_transition_matrix_by_state,
)
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_func_log_crra,
)
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
    # Avoid small values. This test ist just numeric if our solve
    # final period is doing the right thing!
    savings_grid = np.linspace(100, max_wealth + 100, n_grid_points)

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

    if np.allclose(params_dict["theta"], 1):
        util_func = utiility_func_log_crra
    else:
        util_func = utility_func_crra

    user_utility_functions = {
        "utility": util_func,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": marginal_utility_crra,  # Doesn't matter here
    }

    (
        compute_utility,
        compute_marginal_utility,
        compute_inverse_marginal_utility,
        compute_value,
        compute_next_period_wealth,
        transition_vector_by_state,
    ) = get_partial_functions(
        params_dict=params_dict,
        options=options,
        user_utility_functions=user_utility_functions,
        user_budget_constraint=budget_constraint,
        exogenous_transition_function=get_transition_matrix_by_state,
    )

    policy_final, value_final = final_period_wrapper(
        final_period_states=states_final_period,
        savings_grid=savings_grid,
        options=options,
        compute_utility=compute_utility,
        final_period_solution=solve_final_period,
        choices_child=np.array([0, 1]),
        compute_next_period_wealth=compute_next_period_wealth,
        compute_marginal_utility=compute_marginal_utility,
        compute_value=compute_value,
        taste_shock_scale=params_dict["lambda"],
        exogenous_savings_grid=savings_grid,
        income_shock_draws=np.array([0, 0, 0]),
        income_shock_weights=np.array([0, 0, 0]),
    )

    policy_final_expected = np.tile(savings_grid, (2, 1))

    for state_index in range(n_states):
        for choice in range(n_choices):
            expected_value = vmap(compute_utility, in_axes=(0, None))(
                savings_grid, choice
            )

            aaae(
                policy_final[state_index, choice, :],
                policy_final_expected,
            )
            aaae(
                value_final[state_index, choice, 1],
                expected_value,
            )
