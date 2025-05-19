"""Tests for simulation of consumption-retirement model with exogenous processes."""

import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm.toy_models as toy_models
from dcegm.backward_induction import get_solve_func_for_model
from dcegm.pre_processing.check_options import check_model_config_and_process
from dcegm.pre_processing.setup_model import create_model_dict
from dcegm.simulation.random_keys import draw_random_keys_for_seed
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import (
    simulate_all_periods,
    simulate_final_period,
    simulate_single_period,
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

    model_functions = toy_models.load_example_model_functions("with_exog_ltc")

    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_exog_ltc")
    )

    model = create_model_dict(
        model_config=model_config,
        model_specs=model_specs,
        **model_functions,
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

    initial_states = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
    }
    initial_wealth = np.ones(n_agents) * 10
    initial_states_and_wealth = initial_states, initial_wealth

    # Draw the random keys
    sim_keys, last_period_sim_keys = draw_random_keys_for_seed(
        n_agents=n_agents,
        n_periods=n_periods,
        taste_shock_scale_is_scalar=model["model_funcs"]["taste_shock_function"][
            "taste_shock_scale_is_scalar"
        ],
        seed=seed,
    )

    return {
        "initial_states": initial_states,
        "initial_wealth": initial_wealth,
        "initial_states_and_wealth": initial_states_and_wealth,
        "sim_keys": sim_keys,
        "last_period_sim_keys": last_period_sim_keys,
        "seed": seed,
        "value": value,
        "policy": policy,
        "endog_grid": endog_grid,
        "model": model,
        "params": params,
        "model_config": model_config,
        "model_specs": model_specs,
    }


def test_simulate_lax_scan(model_setup):
    model_structure = model_setup["model"]["model_structure"]
    model_funcs = model_setup["model"]["model_funcs"]

    discrete_states_names = model_structure["discrete_states_names"]
    map_state_choice_to_index = model_structure["map_state_choice_to_index_with_proxy"]

    value = model_setup["value"]
    policy = model_setup["policy"]
    endog_grid = model_setup["endog_grid"]

    states_initial = model_setup["initial_states"]
    wealth_initial = model_setup["initial_wealth"]

    state_space_dict = model_structure["state_space_dict"]
    states_initial = {
        key: value.astype(state_space_dict[key].dtype)
        for key, value in states_initial.items()
    }
    initial_states_and_wealth = states_initial, wealth_initial

    model_config_processed = check_model_config_and_process(model_setup["model_config"])
    choice_range = model_config_processed["choices"]

    simulate_body = partial(
        simulate_single_period,
        params=model_setup["params"],
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model_funcs_sim=model_funcs,
        compute_utility=model_funcs["compute_utility"],
        model_structure_sol=model_structure,
        model_config=model_config_processed,
    )

    # a) lax.scan
    (
        lax_states_and_wealth_beginning_of_final_period,
        lax_sim_dict_zero,
    ) = jax.lax.scan(
        f=simulate_body,
        init=initial_states_and_wealth,
        xs=model_setup["sim_keys"],
    )

    sim_keys_0 = {
        key: model_setup["sim_keys"][key][0, ...]
        for key in model_setup["sim_keys"].keys()
    }

    # b) single call
    (
        states_and_wealth_beginning_of_final_period,
        sim_dict_zero,
    ) = simulate_body(initial_states_and_wealth, sim_keys=sim_keys_0)

    lax_final_period_dict = simulate_final_period(
        lax_states_and_wealth_beginning_of_final_period,
        sim_keys=model_setup["last_period_sim_keys"],
        params=model_setup["params"],
        discrete_states_names=discrete_states_names,
        choice_range=choice_range,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        taste_shock_function=model_funcs["taste_shock_function"],
        compute_utility_final=model_funcs["compute_utility_final"],
    )
    final_period_dict = simulate_final_period(
        states_and_wealth_beginning_of_final_period,
        sim_keys=model_setup["last_period_sim_keys"],
        params=model_setup["params"],
        discrete_states_names=discrete_states_names,
        choice_range=choice_range,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        taste_shock_function=model_funcs["taste_shock_function"],
        compute_utility_final=model_funcs["compute_utility_final"],
    )

    aaae(np.squeeze(lax_sim_dict_zero["taste_shocks"]), sim_dict_zero["taste_shocks"])

    for key in final_period_dict.keys():
        aaae(
            lax_final_period_dict["taste_shocks"],
            final_period_dict["taste_shocks"],
        )
        aaae(lax_final_period_dict[key], final_period_dict[key])


def test_simulate(model_setup):

    value = model_setup["value"]
    policy = model_setup["policy"]
    endog_grid = model_setup["endog_grid"]

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
        n_periods=model_setup["model_config"]["n_periods"],
        params=model_setup["params"],
        seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model=model_setup["model"],
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


def test_simulate_second_continuous_choice(model_setup):
    model_functions_cont = toy_models.load_example_model_functions("with_cont_exp")
    model_functions_ltc = toy_models.load_example_model_functions("with_exog_ltc")

    model_config_cont = model_setup["model_config"].copy()
    model_specs_cont = model_setup["model_specs"].copy()

    model_config_cont["continuous_states"]["experience"] = jnp.linspace(0, 1, 6)
    model_specs_cont["max_init_experience"] = 1

    model_cont = create_model_dict(
        model_config=model_config_cont,
        model_specs=model_specs_cont,
        state_space_functions=model_functions_cont["state_space_functions"],
        utility_functions=model_functions_cont["utility_functions"],
        utility_functions_final_period=model_functions_cont[
            "utility_functions_final_period"
        ],
        budget_constraint=model_functions_ltc["budget_constraint"],
        exogenous_states_transition=model_functions_ltc["exogenous_states_transition"],
    )

    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, shape=(24, 6, 120)) * 0

    value = jnp.repeat(model_setup["value"][:, None, :], 6, axis=1) + noise
    policy = jnp.repeat(model_setup["policy"][:, None, :], 6, axis=1) + noise
    endog_grid = jnp.repeat(model_setup["endog_grid"][:, None, :], 6, axis=1) + noise

    n_agents = 100_000

    states_initial = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
        "experience": np.ones(n_agents),
    }
    wealth_initial = np.ones(n_agents) * 10

    result = simulate_all_periods(
        states_initial=states_initial,
        wealth_initial=wealth_initial,
        n_periods=model_config_cont["n_periods"],
        params=model_setup["params"],
        seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model=model_cont,
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
