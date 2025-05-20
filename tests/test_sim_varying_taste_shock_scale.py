"""Tests for simulation of consumption-retirement model with exogenous processes."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models


def _create_test_objects_from_df(df, params):
    _cond = [df["choice"] == 0, df["choice"] == 1]
    _val = [df["taste_shocks_0"], df["taste_shocks_1"]]
    df["taste_shock_selected_choice"] = np.select(_cond, _val)

    value_period_zero = (
        df.xs(0, level=0)["utility"].mean()
        + model_funcs["read_funcs"]["discount_factor"]
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

    model = dcegm.setup_model(
        model_specs=model_specs, model_config=model_config, **ltc_model_functions
    )
    seed = 111
    n_agents = 100_000

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
    }


def test_simulate(model_setup):

    df = model_setup["df"]

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
