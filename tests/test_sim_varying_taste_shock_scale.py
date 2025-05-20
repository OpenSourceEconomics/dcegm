"""Tests for simulation of consumption-retirement model with exogenous processes."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
<<<<<<< HEAD
from dcegm.backward_induction import get_solve_func_for_model
from dcegm.pre_processing.setup_model import create_model_dict
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import (
    simulate_all_periods,
)
=======
>>>>>>> 83037d3d4520f2db5a2ecf22020ce1ea3851e7b8


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
    ltc_model_functions = toy_models.load_example_model_functions("with_stochastic_ltc")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_stochastic_ltc")
    )

    shock_functions = {"taste_shock_scale_per_state": taste_shock_per_lagged_choice}
    ltc_model_functions["shock_functions"] = shock_functions

<<<<<<< HEAD
    model = create_model_dict(
        model_specs=model_specs, model_config=model_config, **ltc_model_functions
    )

    (
        value,
        policy,
        endog_grid,
    ) = get_solve_func_for_model(
        model
    )(params)

    seed = 111
    n_agents = 1_000
    n_periods = model_config["n_periods"]
=======
    model = dcegm.setup_model(
        model_specs=model_specs, model_config=model_config, **ltc_model_functions
    )
    seed = 111
    n_agents = 100_000
>>>>>>> 83037d3d4520f2db5a2ecf22020ce1ea3851e7b8

    initial_states = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
        "assets_begin_of_period": np.ones(n_agents) * 10,
    }

    df = model.get_solve_and_simulate_func(states_initial=initial_states, seed=seed)(
        params=params
    )

    return {
        "df": df,
        "params": params,
<<<<<<< HEAD
        "model_config": model_config,
=======
>>>>>>> 83037d3d4520f2db5a2ecf22020ce1ea3851e7b8
    }


def test_simulate(model_setup):

<<<<<<< HEAD
    value = model_setup["value"]
    policy = model_setup["policy"]
    endog_grid = model_setup["endog_grid"]
    params = model_setup["params"]
    model_config = model_setup["model_config"]

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
        n_periods=model_config["n_periods"],
        params=params,
        seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model=model_setup["model"],
    )

    df = create_simulation_df(result)
=======
    df = model_setup["df"]
>>>>>>> 83037d3d4520f2db5a2ecf22020ce1ea3851e7b8

    value_period_zero, expected = _create_test_objects_from_df(
        df, model_setup["params"]
    )
    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )

    assert len(ids_violating_absorbing_retirement) == 0
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)


def taste_shock_per_lagged_choice(lagged_choice, params):
    return lagged_choice * 0.5 + (1 - lagged_choice) * params["taste_shock_scale"]
