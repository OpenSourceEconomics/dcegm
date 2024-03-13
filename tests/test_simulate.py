"""Tests for simulation of consumption-retirement model with exogenous processes."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.simulation.simulate import simulate_all_periods_for_model
from dcegm.simulation.simulate import simulate_final_period
from dcegm.simulation.simulate import simulate_single_period
from numpy.testing import assert_array_almost_equal as aaae


def _create_test_objects_from_df(df, params):
    _cond = [df["choice"] == 0, df["choice"] == 1]
    _val = [df["taste_shock_0"], df["taste_shock_1"]]
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
        df.xs(0, level=0)["value"].mean()
        - df.xs(0, level=0)["taste_shock_selected_choice"].mean()
    )

    return value_period_zero, expected


@pytest.fixture()
def model_setup(toy_model_exog_ltc):
    params = toy_model_exog_ltc["params"]
    options = toy_model_exog_ltc["options"]

    state_space_names = toy_model_exog_ltc["state_space_names"]
    value = toy_model_exog_ltc["value"]
    policy_left = toy_model_exog_ltc["policy_left"]
    policy_right = toy_model_exog_ltc["policy_right"]
    endog_grid = toy_model_exog_ltc["endog_grid"]
    exog_state_mapping = toy_model_exog_ltc["exog_state_mapping"]
    get_next_period_state = toy_model_exog_ltc["get_next_period_state"]

    model_funcs = toy_model_exog_ltc["model_funcs"]
    map_state_choice_to_index = toy_model_exog_ltc["map_state_choice_to_index"]

    seed = 111
    n_agents = 1_000_000
    n_periods = options["state_space"]["n_periods"]

    initial_states = {
        "period": np.zeros(n_agents, dtype=np.int16),
        "lagged_choice": np.zeros(
            n_agents, dtype=np.int16
        ),  # all agents start as workers
        "married": np.zeros(n_agents, dtype=np.int16),
        "ltc": np.zeros(n_agents, dtype=np.int16),
    }
    initial_resources = np.ones(n_agents) * 10
    initial_states_and_resources = initial_states, initial_resources

    n_keys = len(initial_resources) + 2
    sim_specific_keys = jnp.array(
        [
            jax.random.split(jax.random.PRNGKey(seed + period), num=n_keys)
            for period in range(n_periods)
        ]
    )

    return {
        "params": params,
        "options": options,
        "state_space_names": state_space_names,
        "value": value,
        "policy_left": policy_left,
        "policy_right": policy_right,
        "endog_grid": endog_grid,
        "exog_state_mapping": exog_state_mapping,
        "model_funcs": model_funcs,
        "map_state_choice_to_index": map_state_choice_to_index,
        "get_next_period_state": get_next_period_state,
        "initial_states": initial_states,
        "initial_resources": initial_resources,
        "initial_states_and_resources": initial_states_and_resources,
        "sim_specific_keys": sim_specific_keys,
        "seed": seed,
    }


def test_simulate_lax_scan(model_setup):
    params = model_setup["params"]
    options = model_setup["options"]
    choice_range = jnp.arange(options["model_params"]["n_choices"])

    state_space_names = model_setup["state_space_names"]
    value = model_setup["value"]
    policy_left = model_setup["policy_left"]
    policy_right = model_setup["policy_right"]
    endog_grid = model_setup["endog_grid"]
    exog_state_mapping = model_setup["exog_state_mapping"]
    get_next_period_state = model_setup["get_next_period_state"]
    model_funcs = model_setup["model_funcs"]
    map_state_choice_to_index = model_setup["map_state_choice_to_index"]

    initial_states_and_resources = (
        model_setup["initial_states"],
        model_setup["initial_resources"],
    )
    sim_specific_keys = model_setup["sim_specific_keys"]

    simulate_body = partial(
        simulate_single_period,
        params=params,
        state_space_names=state_space_names,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_left_solved=policy_left,
        policy_right_solved=policy_right,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        choice_range=jnp.arange(map_state_choice_to_index.shape[-1], dtype=jnp.int16),
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        compute_utility=model_funcs["compute_utility"],
        compute_beginning_of_period_resources=model_funcs[
            "compute_beginning_of_period_resources"
        ],
        exog_state_mapping=exog_state_mapping,
        get_next_period_state=get_next_period_state,
    )

    # lax.scan
    (
        lax_states_and_resources_beginning_of_final_period,
        lax_sim_dict_zero,
    ) = jax.lax.scan(
        f=simulate_body,
        init=initial_states_and_resources,
        xs=sim_specific_keys[:-1],
    )

    # single call
    (
        states_and_resources_beginning_of_final_period,
        sim_dict_zero,
    ) = simulate_body(
        initial_states_and_resources, sim_specific_keys=sim_specific_keys[0]
    )

    lax_final_period_dict = simulate_final_period(
        lax_states_and_resources_beginning_of_final_period,
        sim_specific_keys=sim_specific_keys[-1],
        params=params,
        state_space_names=state_space_names,
        choice_range=choice_range,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )
    final_period_dict = simulate_final_period(
        states_and_resources_beginning_of_final_period,
        sim_specific_keys=sim_specific_keys[-1],
        params=params,
        state_space_names=state_space_names,
        choice_range=choice_range,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )

    aaae(np.squeeze(lax_sim_dict_zero["taste_shocks"]), sim_dict_zero["taste_shocks"])

    for key in final_period_dict.keys():
        aaae(
            lax_final_period_dict["taste_shocks"],
            final_period_dict["taste_shocks"],
        )
        aaae(lax_final_period_dict[key], final_period_dict[key])


def test_simulate(model_setup):
    params = model_setup["params"]
    options = model_setup["options"]

    state_space_names = model_setup["state_space_names"]
    value = model_setup["value"]
    policy_left = model_setup["policy_left"]
    policy_right = model_setup["policy_right"]
    endog_grid = model_setup["endog_grid"]
    exog_state_mapping = model_setup["exog_state_mapping"]
    get_next_period_state = model_setup["get_next_period_state"]
    model_funcs = model_setup["model_funcs"]
    map_state_choice_to_index = model_setup["map_state_choice_to_index"]

    initial_states = model_setup["initial_states"]
    initial_resources = model_setup["initial_resources"]

    seed = model_setup["seed"]

    result = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=initial_resources,
        n_periods=options["state_space"]["n_periods"],
        params=params,
        state_space_names=state_space_names,
        seed=seed,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_left_solved=policy_left,
        policy_right_solved=policy_right,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        choice_range=jnp.arange(map_state_choice_to_index.shape[-1], dtype=jnp.int16),
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        compute_utility=model_funcs["compute_utility"],
        compute_beginning_of_period_resources=model_funcs[
            "compute_beginning_of_period_resources"
        ],
        exog_state_mapping=exog_state_mapping,
        get_next_period_state=get_next_period_state,
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )

    df = create_simulation_df(result)

    value_period_zero, expected = _create_test_objects_from_df(df, params)
    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )

    assert len(ids_violating_absorbing_retirement) == 0
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)


def test_simulate_all_periods_for_model(model_setup):
    states_initial = model_setup["initial_states"]
    resources_initial = model_setup["initial_resources"]
    n_periods = model_setup["options"]["state_space"]["n_periods"]
    params = model_setup["params"]
    seed = model_setup["seed"]
    endog_grid_solved = model_setup["endog_grid"]
    value_solved = model_setup["value"]
    policy_left_solved = model_setup["policy_left"]
    policy_right_solved = model_setup["policy_right"]
    choice_range = jnp.arange(
        model_setup["map_state_choice_to_index"].shape[-1], dtype=jnp.int16
    )
    model = {
        "state_space_names": model_setup["state_space_names"],
        "map_state_choice_to_index": model_setup["map_state_choice_to_index"],
        "model_funcs": model_setup["model_funcs"],
        "exog_mapping": model_setup["exog_state_mapping"],
        "get_next_period_state": model_setup["get_next_period_state"],
    }

    result = simulate_all_periods_for_model(
        states_initial=states_initial,
        resources_initial=resources_initial,
        n_periods=n_periods,
        params=params,
        seed=seed,
        endog_grid_solved=endog_grid_solved,
        value_solved=value_solved,
        policy_left_solved=policy_left_solved,
        policy_right_solved=policy_right_solved,
        choice_range=choice_range,
        model=model,
    )

    df = create_simulation_df(result)

    value_period_zero, expected = _create_test_objects_from_df(df, params)
    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )

    assert len(ids_violating_absorbing_retirement) == 0
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)
