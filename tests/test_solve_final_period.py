from functools import partial
from itertools import product

import numpy as np
import pytest
from dcegm.final_period import solve_final_period
from dcegm.pre_processing import convert_params_to_dict
from dcegm.pre_processing import get_partial_functions
from dcegm.state_space import create_state_choice_space
from jax import vmap
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.exogenous_processes import (
    get_transition_matrix_by_state,
)
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period_scalar,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

model = ["deaton", "retirement_taste_shocks", "retirement_no_taste_shocks"]
model = ["retirement_taste_shocks", "retirement_no_taste_shocks"]
max_wealth = [50, 11]
n_grid_points = [1000, 101]
TEST_CASES = list(product(model, max_wealth, n_grid_points))


@pytest.mark.parametrize("model, max_wealth, n_grid_points", TEST_CASES)
def test_consume_everything_in_final_period(
    model, max_wealth, n_grid_points, load_example_model
):
    params, options = load_example_model(f"{model}")
    params_dict = convert_params_to_dict(params)
    options["n_exog_states"] = 1
    options["n_discrete_choices"]
    n_periods = options["n_periods"]

    # Avoid small values. This test is just numeric if our solve
    # final period is doing the right thing!
    savings_grid = np.linspace(100, max_wealth + 100, n_grid_points)

    state_space, state_indexer = create_state_space(options)
    (
        state_choice_space,
        map_state_choice_vec_to_parent_state,
        _reshape_state_choice_vec_to_mat,
        transform_between_state_and_state_choice_space,
    ) = create_state_choice_space(
        state_space, state_indexer, get_state_specific_feasible_choice_set
    )
    idx_states_final_period = np.where(state_space[:, 0] == n_periods - 1)[0]

    idx_state_choice_combs = np.where(state_choice_space[:, 0] == n_periods - 1)[0]
    final_period_state_choice_combs = state_choice_space[idx_state_choice_combs]
    map_final_state_choice_to_state = map_state_choice_vec_to_parent_state[
        idx_state_choice_combs
    ]

    user_utility_functions = {
        "utility": utility_func_crra,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": marginal_utility_crra,  # Doesn't matter here
    }

    (
        compute_utility,
        compute_marginal_utility,
        _compute_inverse_marginal_utility,
        _compute_value,
        compute_next_period_wealth,
        _compute_upper_envelope,
        _transition_vector_by_state,
    ) = get_partial_functions(
        params_dict=params_dict,
        options=options,
        user_utility_functions=user_utility_functions,
        user_budget_constraint=budget_constraint,
        exogenous_transition_function=get_transition_matrix_by_state,
    )

    income_draws = np.array([0, 0, 0])

    resources_beginning_of_period = vmap(
        vmap(
            vmap(compute_next_period_wealth, in_axes=(None, None, 0, None)),
            in_axes=(None, 0, None, None),
        ),
        in_axes=(0, None, None, None),
    )(state_space, savings_grid, income_draws, params_dict)

    (
        transform_between_state_and_state_choice_space[idx_states_final_period, :][
            :, idx_state_choice_combs
        ]
    )

    resources_last_period = resources_beginning_of_period[
        map_final_state_choice_to_state
    ]

    final_period_solution_partial = partial(
        solve_final_period_scalar,
        options=options,
        compute_utility=compute_utility,
        compute_marginal_utility=compute_marginal_utility,
    )

    _, value_final, policy_final = solve_final_period(
        state_choice_mat=final_period_state_choice_combs,
        resources=resources_last_period,
        final_period_solution_partial=final_period_solution_partial,
        params=params_dict,
    )

    for state_choice_idx, state_choice in enumerate(final_period_state_choice_combs):
        state = state_choice[:-1]
        choice = state_choice[-1]
        begin_of_period_resources = vmap(
            compute_next_period_wealth, in_axes=(None, 0, None, None)
        )(state, savings_grid, 0.0, params_dict)

        aaae(
            resources_last_period[state_choice_idx, :, 1],
            begin_of_period_resources,
        )

        aaae(
            policy_final[state_choice_idx, :, 1],
            begin_of_period_resources,
        )

        expected_value = vmap(compute_utility, in_axes=(0, None, None))(
            begin_of_period_resources, choice, params_dict
        )
        aaae(
            value_final[state_choice_idx, :, 1],
            expected_value,
        )
