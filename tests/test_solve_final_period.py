from itertools import product

import numpy as np
import pytest
from dcegm.final_period import final_period_wrapper
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
    get_state_specific_choice_set,
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
    options["n_exog_processes"] = 1
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]

    # Avoid small values. This test is just numeric if our solve
    # final period is doing the right thing!
    savings_grid = np.linspace(100, max_wealth + 100, n_grid_points)

    state_space, state_indexer = create_state_space(options)
    state_choice_space = create_state_choice_space(
        state_space, state_indexer, get_state_specific_choice_set
    )

    states_final_period = state_space[np.where(state_space[:, 0] == n_periods - 1)]

    idx_state_choice_combs = np.where(state_choice_space[:, 0] == n_periods - 1)[0]
    feasible_state_choice_combs = state_choice_space[idx_state_choice_combs]

    choices_child_states = np.zeros(
        (state_space.shape[0] // n_periods, n_choices), dtype=int
    )

    for state_choice_idx, state_choice_vec in enumerate(feasible_state_choice_combs):
        choices_child_states[state_choice_idx - n_choices, state_choice_vec[-1]] = True

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
            vmap(compute_next_period_wealth, in_axes=(None, None, 0)),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, None, None),
    )(states_final_period, savings_grid, income_draws)

    endog_grid_final, policy_final, value_final, *_ = final_period_wrapper(
        final_period_states=states_final_period,
        options=options,
        compute_utility=compute_utility,
        final_period_solution=solve_final_period_scalar,
        choices_final=choices_child_states,
        compute_marginal_utility=compute_marginal_utility,
        taste_shock_scale=params_dict["lambda"],
        resources_last_period=resources_beginning_of_period,
        income_shock_draws=income_draws,
        income_shock_weights=np.array([0, 0, 0]),
    )

    for state_idx, state in enumerate(states_final_period):
        for choice in range(n_choices):
            begin_of_period_resources = vmap(
                compute_next_period_wealth, in_axes=(None, 0, None)
            )(state, savings_grid, 0.00)

            aaae(
                endog_grid_final[state_idx, choice],
                begin_of_period_resources,
            )

            aaae(
                policy_final[state_idx, choice],
                begin_of_period_resources,
            )

            expected_value = vmap(compute_utility, in_axes=(0, None))(
                begin_of_period_resources, choice
            )
            aaae(
                value_final[state_idx, choice],
                expected_value,
            )
