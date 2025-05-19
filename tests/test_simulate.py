"""Tests for simulation of consumption-retirement model with exogenous processes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import (
    simulate_all_periods,
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

    model_functions = toy_models.load_example_model_functions("with_stochastic_ltc")

    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_stochastic_ltc")
    )

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_functions,
    )

    model_solved = model.solve(params)

    return {
        "seed": 111,
        "model_solved": model_solved,
        "model": model,
        "params": params,
        "model_config": model_config,
        "model_specs": model_specs,
    }


def test_simulate(model_setup):
    n_agents = 100_000

    discrete_initial_states = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
        "assets_begin_of_period": np.ones(n_agents) * 10,
    }

    df = model_setup["model_solved"].simulate(
        states_initial=discrete_initial_states, seed=111
    )

    value_period_zero, expected = _create_test_objects_from_df(
        df, model_setup["params"]
    )
    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )

    assert len(ids_violating_absorbing_retirement) == 0
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)


def test_simulate_second_continuous_choice(model_setup):
    model_functions_cont = toy_models.load_example_model_functions("with_cont_exp")
    model_functions_ltc = toy_models.load_example_model_functions("with_stochastic_ltc")

    model_config_cont = model_setup["model_config"].copy()
    model_specs_cont = model_setup["model_specs"].copy()

    model_config_cont["continuous_states"]["experience"] = jnp.linspace(0, 1, 6)
    model_specs_cont["max_init_experience"] = 1

    model_cont = dcegm.setup_model(
        model_config=model_config_cont,
        model_specs=model_specs_cont,
        state_space_functions=model_functions_cont["state_space_functions"],
        utility_functions=model_functions_cont["utility_functions"],
        utility_functions_final_period=model_functions_cont[
            "utility_functions_final_period"
        ],
        budget_constraint=model_functions_ltc["budget_constraint"],
        stochastic_states_transitions=model_functions_ltc[
            "stochastic_states_transitions"
        ],
    )

    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, shape=(24, 6, 120)) * 0
    model_solved = model_setup["model_solved"]

    value = jnp.repeat(model_solved.value[:, None, :], 6, axis=1) + noise
    policy = jnp.repeat(model_solved.policy[:, None, :], 6, axis=1) + noise
    endog_grid = jnp.repeat(model_solved.endog_grid[:, None, :], 6, axis=1) + noise

    n_agents = 100_000

    states_initial = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
        "experience": np.ones(n_agents),
        "assets_begin_of_period": np.ones(n_agents) * 10,
    }

    result = simulate_all_periods(
        states_initial=states_initial,
        n_periods=model_config_cont["n_periods"],
        params=model_setup["params"],
        seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model_funcs=model_cont.model_funcs,
        model_config=model_cont.model_config,
        model_structure=model_cont.model_structure,
    )

    df = create_simulation_df(result)

    value_period_zero, expected = _create_test_objects_from_df(
        df, model_setup["params"]
    )
    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )

    assert len(ids_violating_absorbing_retirement) == 0
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)
