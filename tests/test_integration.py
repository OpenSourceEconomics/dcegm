import pickle
from pathlib import Path

import numpy as np
import pytest
from dcegm.solve import solve_dcegm
from dcegm.state_space import create_state_choice_space
from jax.config import config
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
    inverse_marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    utiility_func_log_crra,
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


@pytest.mark.parametrize(
    "model, choice_range",
    [
        ("retirement_no_taste_shocks", [0, 1]),
        ("retirement_taste_shocks", [0, 1]),
        ("deaton", [0]),
    ],
)
def test_benchmark_models(
    model,
    choice_range,
    utility_functions,
    state_space_functions,
    load_example_model,
):
    params, options = load_example_model(f"{model}")
    options["n_exog_processes"] = 1

    state_space, map_state_to_index = create_state_space(options)
    (
        state_choice_space,
        sum_state_choices_to_state,
        map_state_choice_to_state,
        _,
    ) = create_state_choice_space(
        state_space,
        map_state_to_index,
        state_space_functions["get_state_specific_choice_set"],
    )

    if params.loc[("utility_function", "theta"), "value"] == 1:
        utility_functions["utility"] = utiility_func_log_crra

    endog_grid_calculated, policy_calculated, value_calculated = solve_dcegm(
        params,
        options,
        utility_functions,
        budget_constraint=budget_constraint,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
        transition_function=get_transition_matrix_by_state,
    )

    policy_expected = pickle.load(
        (TEST_RESOURCES_DIR / f"policy_{model}.pkl").open("rb")
    )
    value_expected = pickle.load((TEST_RESOURCES_DIR / f"value_{model}.pkl").open("rb"))

    for period in range(23, -1, -1):
        state_choices_ids_period = np.where(state_choice_space[:, 0] == period)[0]

        for state_choice_idx in state_choices_ids_period:
            choice = state_choice_space[state_choice_idx, -1]
            if model == "deaton":
                policy_expec = policy_expected[period, choice]
                value_expec = value_expected[period, choice]
            else:
                policy_expec = policy_expected[period][1 - choice].T
                value_expec = value_expected[period][1 - choice].T

            endog_grid_got = endog_grid_calculated[state_choice_idx][
                ~np.isnan(endog_grid_calculated[state_choice_idx]),
            ]

            aaae(endog_grid_got, policy_expec[0])

            policy_got = policy_calculated[state_choice_idx][
                ~np.isnan(policy_calculated[state_choice_idx]),
            ]
            aaae(policy_got, policy_expec[1])

            # In Fedor's upper envelope, there are two endogenous wealth grids;
            # one for the value function and a longer one for the policy function.
            # Since we want to unify the two endogoenous grids and want the refined
            # value and policy array to be of equal length, our refined value
            # function is longer than Fedor's.
            # Hence, we interpolate Fedor's refined value function to our refined
            # grid.
            value_expec_interp = np.interp(
                policy_expec[0], value_expec[0], value_expec[1]
            )
            value_got = value_calculated[state_choice_idx][
                ~np.isnan(value_calculated[state_choice_idx])
            ]

            aaae(value_got, value_expec_interp)
