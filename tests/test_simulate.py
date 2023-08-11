from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dcegm.interpolation import linear_interpolation_with_inserting_missing_values
from dcegm.pre_processing import calc_current_value
from dcegm.pre_processing import convert_params_to_dict
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
from toy_models.consumption_retirement_model.simulate import simulate_stacked
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
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


def _interpolate_value_on_current_grid(
    choice, wealth, endog_grid, value, beta, calc_current_value, compute_utility
):
    wealth = wealth.flatten("F")
    value_interp = np.full_like(wealth, np.nan)

    endog_grid_min = endog_grid[0]
    value_min = value[0]
    credit_constraint = wealth < endog_grid_min

    value_interp[
        ~credit_constraint
    ] = linear_interpolation_with_inserting_missing_values(
        x=endog_grid,
        y=value,
        x_new=wealth[~credit_constraint],
        missing_value=np.nan,
    )

    value_interp_closed_form = calc_current_value(
        consumption=wealth,
        next_period_value=value_min,
        choice=choice,
        discount_factor=beta,
        compute_utility=compute_utility,
    )
    value_interp[credit_constraint] = value_interp_closed_form[credit_constraint]

    return value_interp


@pytest.fixture()
def state_space_functions():
    """Return dict with utility functions."""
    return {
        "create_state_space": create_state_space,
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
    }


def test_simulate(utility_functions, state_space_functions, load_example_model):
    model = "retirement_no_taste_shocks"

    params, options = load_example_model(f"{model}")
    options["n_exog_processes"] = 1
    params.loc[("shocks", "lambda"), "value"] = 2.2204e-16
    params.loc[("assets", "initial_wealth_low"), "value"] = 10
    params.loc[("assets", "initial_wealth_high"), "value"] = 30
    params_dict = convert_params_to_dict(params)

    compute_utility = partial(utility_func_crra, params_dict=params_dict)
    interpolate_value_on_current_grid = partial(
        _interpolate_value_on_current_grid,
        beta=params.loc[("beta", "beta"), "value"],
        calc_current_value=calc_current_value,
        compute_utility=compute_utility,
    )

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

    # TODO: Fix final period solution in `solve.py`. For now, do manually
    _max_wealth = params.loc[("assets", "max_wealth"), "value"]
    _idx_possible_states = np.where(state_space[:, 0] == options["n_periods"] - 1)[0]

    endog_grid[_idx_possible_states, ...] = np.nan
    policy[_idx_possible_states, ...] = np.nan
    value[_idx_possible_states, ...] = np.nan
    endog_grid[_idx_possible_states, ..., :2] = [0, _max_wealth]
    policy[_idx_possible_states, ..., :2] = [0, _max_wealth]
    value[_idx_possible_states, ..., :2] = [0, _max_wealth]

    df = simulate_stacked(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        # discount_factor=params.loc[("beta", "beta"), "value"],
        # delta=params.loc[("delta", "delta"), "value"],
        # theta=params.loc[("utility_function", "theta"), "value"],
        coeffs_age_poly=params.loc[("wage"), "value"],
        wage_shock_scale=params.loc[("shocks", "sigma"), "value"],
        taste_shock_scale=params.loc[("shocks", "lambda"), "value"],
        initial_wealth=[
            params.loc[("assets", "initial_wealth_low"), "value"],
            params.loc[("assets", "initial_wealth_high"), "value"],
        ],
        interest_rate=params.loc[("assets", "interest_rate"), "value"],
        n_periods=options["n_periods"],
        state_space=state_space,
        indexer=indexer,
        interpolate_value_on_current_grid=interpolate_value_on_current_grid,
        n_sims=100,
        seed=7134,
    )

    expected = pd.read_csv(TEST_RESOURCES_DIR / "df_simulate_fedor.csv")
    aaae(df, expected)
