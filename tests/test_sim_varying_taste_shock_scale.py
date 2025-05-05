"""Tests for simulation of consumption-retirement model with exogenous processes."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.pre_processing.setup_model import setup_model
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import (
    simulate_all_periods,
)
from dcegm.solve import get_solve_func_for_model
from tests.test_models.exog_ltc_model import OPTIONS, PARAMS, budget_dcegm_exog_ltc
from toy_models.cons_ret_model_dcegm_paper.state_space_objects import (
    create_state_space_function_dict,
)
from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)


def _create_test_objects_from_df(df, params):
    _cond = [df["choice"] == 0, df["choice"] == 1]
    _val = [df["taste_shocks_0"], df["taste_shocks_1"]]
    df["taste_shock_selected_choice"] = np.select(_cond, _val)

    value_period_zero = (
        df.xs(0, level=0)["utility"].mean()
        + params["beta"]
        * (
            df.xs(1, level=0)["utility"]
            + df.xs(1, level=0)["taste_shock_selected_choice"]
        ).mean()
    )
    expected = (
        df.xs(0, level=0)["value_max"].mean()
        - df.xs(0, level=0)["taste_shock_selected_choice"].mean()
    )

    return value_period_zero, expected


@pytest.fixture()
def model_setup():
    shock_functions = {"taste_shock_scale_per_state": taste_shock_per_lagged_choice}

    model = setup_model(
        options=OPTIONS,
        state_space_functions=create_state_space_function_dict(),
        utility_functions=create_utility_function_dict(),
        utility_functions_final_period=create_final_period_utility_function_dict(),
        budget_constraint=budget_dcegm_exog_ltc,
        shock_functions=shock_functions,
    )

    (
        value,
        policy,
        endog_grid,
    ) = get_solve_func_for_model(
        model
    )(PARAMS)

    seed = 111
    n_agents = 1_000
    n_periods = OPTIONS["state_space"]["n_periods"]

    initial_states = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
    }
    initial_wealth = np.ones(n_agents) * 10
    initial_states_and_wealth = initial_states, initial_wealth

    n_keys = len(initial_wealth) + 2
    sim_specific_keys = jnp.array(
        [
            jax.random.split(jax.random.PRNGKey(seed + period), num=n_keys)
            for period in range(n_periods)
        ]
    )

    return {
        "initial_states": initial_states,
        "initial_wealth": initial_wealth,
        "initial_states_and_wealth": initial_states_and_wealth,
        "sim_specific_keys": sim_specific_keys,
        "seed": seed,
        "value": value,
        "policy": policy,
        "endog_grid": endog_grid,
        "model": model,
        "params": PARAMS.copy(),
        "options": OPTIONS.copy(),
    }


def test_simulate(model_setup):

    value = model_setup["value"]
    policy = model_setup["policy"]
    endog_grid = model_setup["endog_grid"]
    params = model_setup["params"]
    options = model_setup["options"]

    n_agents = 100_000

    discrete_initial_states = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
    }
    wealth_initial = np.ones(n_agents) * 10

    result = simulate_all_periods(
        states_initial=discrete_initial_states,
        wealth_initial=wealth_initial,
        n_periods=options["state_space"]["n_periods"],
        params=params,
        seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model=model_setup["model"],
    )

    df = create_simulation_df(result)

    value_period_zero, expected = _create_test_objects_from_df(df, PARAMS)
    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )

    assert len(ids_violating_absorbing_retirement) == 0
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)


def taste_shock_per_lagged_choice(lagged_choice, params):
    return lagged_choice * 0.5 + (1 - lagged_choice) * params["taste_shock_scale"]
