"""The simulation function."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from dcegm.interfaces.interface import get_state_choice_index_per_discrete_state
from dcegm.simulation.random_keys import draw_random_keys_for_seed
from dcegm.simulation.sim_utils import (
    compute_final_utility_for_each_choice,
    interpolate_policy_and_value_for_all_agents,
    transition_to_next_period,
    vectorized_utility,
)
from dcegm.simulation.taste_shocks import draw_taste_shocks


def simulate_all_periods(
    states_initial,
    wealth_initial,
    n_periods,
    params,
    seed,
    endog_grid_solved,
    policy_solved,
    value_solved,
    model,
    alt_model_funcs_sim=None,
):
    alt_model_funcs_sim = (
        model["model_funcs"] if alt_model_funcs_sim is None else alt_model_funcs_sim
    )

    model_structure_sol = model["model_structure"]
    discrete_state_space = model_structure_sol["state_space_dict"]
    model_funcs_sol = model["model_funcs"]
    model_config_sol = model["model_config"]

    # Set initial states to internal dtype
    states_initial_dtype = {
        key: value.astype(discrete_state_space[key].dtype)
        for key, value in states_initial.items()
        if key in discrete_state_space
    }

    if "dummy_exog" in model_structure_sol["exog_states_names"]:
        states_initial_dtype["dummy_exog"] = jnp.zeros_like(
            states_initial_dtype["period"]
        )

    continuous_states_info = model_config_sol["continuous_states_info"]

    if continuous_states_info["second_continuous_exists"]:
        states_initial_dtype[continuous_states_info["second_continuous_state_name"]] = (
            states_initial[continuous_states_info["second_continuous_state_name"]]
        )

    n_agents = len(wealth_initial)

    # Draw the random keys
    sim_keys, last_period_sim_keys = draw_random_keys_for_seed(
        n_agents=n_agents,
        n_periods=n_periods,
        taste_shock_scale_is_scalar=alt_model_funcs_sim["taste_shock_function"][
            "taste_shock_scale_is_scalar"
        ],
        seed=seed,
    )

    simulate_body = partial(
        simulate_single_period,
        params=params,
        endog_grid_solved=endog_grid_solved,
        value_solved=value_solved,
        policy_solved=policy_solved,
        model_structure_sol=model_structure_sol,
        model_funcs_sim=alt_model_funcs_sim,
        compute_utility=model_funcs_sol["compute_utility"],
        model_config=model_config_sol,
    )

    states_and_wealth_beginning_of_first_period = (
        states_initial_dtype,
        wealth_initial,
    )

    states_and_wealth_beginning_of_final_period, sim_dict = jax.lax.scan(
        f=simulate_body,
        init=states_and_wealth_beginning_of_first_period,
        xs=sim_keys,
    )

    final_period_dict = simulate_final_period(
        states_and_wealth_beginning_of_final_period,
        sim_keys=last_period_sim_keys,
        params=params,
        discrete_states_names=model_structure_sol["discrete_states_names"],
        choice_range=model_structure_sol["choice_range"],
        map_state_choice_to_index=jnp.asarray(
            model_structure_sol["map_state_choice_to_index_with_proxy"]
        ),
        taste_shock_function=alt_model_funcs_sim["taste_shock_function"],
        compute_utility_final=model_funcs_sol["compute_utility_final"],
    )

    # Standard simulation output

    result = {
        key: jnp.vstack([sim_dict[key], final_period_dict[key]])
        for key in sim_dict.keys()
        if key in final_period_dict.keys()
    }
    n_array_agents = jnp.ones(n_agents, dtype=float) * jnp.nan
    aux_results = {
        key: jnp.vstack([n_array_agents, sim_dict[key]])
        for key in sim_dict.keys()
        if key not in final_period_dict.keys()
    }
    result = {**result, **aux_results}
    return result


def simulate_single_period(
    states_and_wealth_beginning_of_period,
    sim_keys,
    params,
    endog_grid_solved,
    value_solved,
    policy_solved,
    model_structure_sol,
    model_funcs_sim,
    compute_utility,
    model_config,
):

    (
        states_beginning_of_period,
        wealth_beginning_of_period,
    ) = states_and_wealth_beginning_of_period

    continuous_states_info = model_config["continuous_states_info"]

    if continuous_states_info["second_continuous_exists"]:
        continuous_state_name = continuous_states_info["second_continuous_state_name"]
        continuous_grid = continuous_states_info["second_continuous_grid"]

        continuous_state_beginning_of_period = states_beginning_of_period[
            continuous_state_name
        ]
        discrete_states_beginning_of_period = {
            key: value
            for key, value in states_beginning_of_period.items()
            if key != continuous_state_name
        }
    else:
        discrete_states_beginning_of_period = states_beginning_of_period
        continuous_state_beginning_of_period = None
        continuous_grid = None

    choice_range = model_structure_sol["choice_range"]
    # Interpolate policy and value function for all agents.
    policy, values_pre_taste_shock = interpolate_policy_and_value_for_all_agents(
        discrete_states_beginning_of_period=discrete_states_beginning_of_period,
        continuous_state_beginning_of_period=continuous_state_beginning_of_period,
        wealth_beginning_of_period=wealth_beginning_of_period,
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        map_state_choice_to_index=jnp.asarray(
            model_structure_sol["map_state_choice_to_index_with_proxy"]
        ),
        choice_range=model_structure_sol["choice_range"],
        params=params,
        discrete_states_names=model_structure_sol["discrete_states_names"],
        compute_utility=compute_utility,
        continuous_grid=continuous_grid,
    )

    # Draw taste shocks and calculate final value.
    taste_shocks = draw_taste_shocks(
        params=params,
        states_beginning_of_period=states_beginning_of_period,
        n_choices=len(choice_range),
        taste_shock_function=model_funcs_sim["taste_shock_function"],
        taste_shock_keys=sim_keys["taste_shock_keys"],
    )
    values_across_choices = values_pre_taste_shock + taste_shocks

    # Determine choice index of period by max value and select corresponding choice,
    # consumption and value.
    choice_index = jnp.nanargmax(values_across_choices, axis=1)
    choice = choice_range[choice_index]

    value_max = jnp.take_along_axis(
        values_across_choices, choice_index[:, None], axis=1
    )[:, 0]

    consumption = jnp.take_along_axis(policy, choice_index[:, None], axis=1)[:, 0]
    utility_period = vmap(vectorized_utility, in_axes=(0, 0, 0, None, None))(
        consumption,
        states_beginning_of_period,
        choice,
        params,
        compute_utility,
    )
    savings_current_period = wealth_beginning_of_period - consumption

    (
        wealth_beginning_of_next_period,
        budget_aux,
        discrete_states_next_period,
        continuous_state_next_period,
        income_shocks_next_period,
    ) = transition_to_next_period(
        discrete_states_beginning_of_period=discrete_states_beginning_of_period,
        continuous_state_beginning_of_period=continuous_state_beginning_of_period,
        savings_current_period=savings_current_period,
        choice=choice,
        params=params,
        model_funcs_sim=model_funcs_sim,
        sim_keys=sim_keys,
    )

    states_next_period = discrete_states_next_period

    if continuous_states_info["second_continuous_exists"]:
        states_next_period[continuous_state_name] = continuous_state_next_period

    carry = states_next_period, wealth_beginning_of_next_period

    result = {
        "choice": choice,
        "consumption": consumption,
        "utility": utility_period,
        "taste_shocks": taste_shocks,
        "value_max": value_max,
        "value_choice": values_across_choices,
        "wealth_beginning_of_period": wealth_beginning_of_period,
        "savings": savings_current_period,
        "income_shock": income_shocks_next_period,
        **budget_aux,
        **states_beginning_of_period,
    }

    return carry, result


def simulate_final_period(
    states_and_wealth_beginning_of_period,
    sim_keys,
    params,
    discrete_states_names,
    choice_range,
    map_state_choice_to_index,
    taste_shock_function,
    compute_utility_final,
):
    invalid_number = np.iinfo(map_state_choice_to_index.dtype).max

    (
        states_beginning_of_final_period,
        wealth_beginning_of_final_period,
    ) = states_and_wealth_beginning_of_period

    n_agents = len(wealth_beginning_of_final_period)

    utilities_pre_taste_shock = vmap(
        vmap(
            compute_final_utility_for_each_choice,
            in_axes=(None, 0, None, None, None),  # choices
        ),
        in_axes=(0, None, 0, None, None),  # agents
    )(
        states_beginning_of_final_period,
        choice_range,
        wealth_beginning_of_final_period,
        params,
        compute_utility_final,
    )
    state_choice_indexes = get_state_choice_index_per_discrete_state(
        map_state_choice_to_index=map_state_choice_to_index,
        states=states_beginning_of_final_period,
        discrete_states_names=discrete_states_names,
    )
    utilities_pre_taste_shock = jnp.where(
        state_choice_indexes == invalid_number, np.nan, utilities_pre_taste_shock
    )

    # Draw taste shocks and calculate final value.
    taste_shocks = draw_taste_shocks(
        params=params,
        states_beginning_of_period=states_beginning_of_final_period,
        n_choices=len(choice_range),
        taste_shock_function=taste_shock_function,
        taste_shock_keys=sim_keys["taste_shock_keys"],
    )
    values_across_choices = utilities_pre_taste_shock + taste_shocks

    choice_index = jnp.nanargmax(values_across_choices, axis=1)
    choice = choice_range[choice_index]

    utility_period = jnp.take_along_axis(
        utilities_pre_taste_shock, choice_index[:, None], axis=1
    )[:, 0]
    value_period = jnp.take_along_axis(
        values_across_choices, choice_index[:, None], axis=1
    )[:, 0]

    result = {
        "choice": choice,
        "consumption": wealth_beginning_of_final_period,
        "utility": utility_period,
        "value_max": value_period,
        "value_choice": values_across_choices[np.newaxis],
        "taste_shocks": taste_shocks[np.newaxis, :, :],
        "wealth_beginning_of_period": wealth_beginning_of_final_period,
        "savings": jnp.zeros_like(utility_period),
        "income_shock": jnp.zeros(n_agents),
        **states_beginning_of_final_period,
    }

    return result
