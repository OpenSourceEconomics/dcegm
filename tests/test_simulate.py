"""Tests for simulation of consumption-retirement model with exogenous processes."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.simulation.simulate import simulate_final_period
from dcegm.simulation.simulate import simulate_single_period
from numpy.testing import assert_array_almost_equal as aaae


def test_simulate_lax_scan(toy_model_exog_ltc):
    params = toy_model_exog_ltc["params"]
    options = toy_model_exog_ltc["options"]
    choice_range = jnp.arange(options["model_params"]["n_choices"])
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
    n_agents = 1_000
    n_periods = options["state_space"]["n_periods"]

    # === Simulate ===

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


def test_simulate(toy_model_exog_ltc):
    params = toy_model_exog_ltc["params"]
    options = toy_model_exog_ltc["options"]
    jnp.arange(options["model_params"]["n_choices"])
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
    n_agents = 100_000

    # === Simulate ===

    initial_states = {
        "period": np.zeros(n_agents, dtype=np.int16),
        "lagged_choice": np.zeros(
            n_agents, dtype=np.int16
        ),  # all agents start as workers
        "married": np.zeros(n_agents, dtype=np.int16),
        "ltc": np.zeros(n_agents, dtype=np.int16),
    }
    initial_resources = np.ones(n_agents) * 10

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

    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )
    assert len(ids_violating_absorbing_retirement) == 0

    aaae(value_period_zero.mean(), expected.mean(), decimal=2)
