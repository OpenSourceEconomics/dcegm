from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from dcegm.simulation.sim_final_period import simulate_final_period
from dcegm.simulation.sim_utils import draw_taste_shocks
from dcegm.simulation.sim_utils import interpolate_policy_and_value_for_all_agents
from dcegm.simulation.sim_utils import transition_to_next_period
from dcegm.simulation.sim_utils import vectorized_utility
from jax import vmap


def simulate_all_periods(
    states_initial,
    resources_initial,
    n_periods,
    params,
    seed,
    endog_grid_solved,
    value_solved,
    policy_left_solved,
    policy_right_solved,
    map_state_choice_to_index,
    choice_range,
    compute_exog_transition_vec,
    compute_utility,
    compute_beginning_of_period_resources,
    exog_state_mapping,
    update_endog_state_by_state_and_choice,
    compute_utility_final_period,
):
    simulate_body = partial(
        simulate_single_period,
        params=params,
        basic_seed=seed,
        endog_grid_solved=endog_grid_solved,
        value_solved=value_solved,
        policy_left_solved=policy_left_solved,
        policy_right_solved=policy_right_solved,
        map_state_choice_to_index=map_state_choice_to_index,
        choice_range=choice_range,
        compute_exog_transition_vec=compute_exog_transition_vec,
        compute_utility=compute_utility,
        compute_beginning_of_period_resources=compute_beginning_of_period_resources,
        exog_state_mapping=exog_state_mapping,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    states_and_resources_beginning_of_first_period = states_initial, resources_initial
    states_and_resources_beginning_of_final_period, sim_dict = jax.lax.scan(
        f=simulate_body,
        init=states_and_resources_beginning_of_first_period,
        xs=jnp.arange(n_periods - 1),
    )

    final_period_dict = simulate_final_period(
        states_and_resources_beginning_of_final_period,
        period=n_periods,
        params=params,
        basic_seed=seed,
        choice_range=choice_range,
        compute_utility_final_period=compute_utility_final_period,
    )

    result = {
        key: np.row_stack([sim_dict[key], final_period_dict[key]])
        for key in sim_dict.keys()
    }

    return result


def simulate_single_period(
    states_and_resources_beginning_of_period,
    period,
    params,
    basic_seed,
    endog_grid_solved,
    value_solved,
    policy_left_solved,
    policy_right_solved,
    map_state_choice_to_index,
    choice_range,
    compute_exog_transition_vec,
    compute_utility,
    compute_beginning_of_period_resources,
    exog_state_mapping,
    update_endog_state_by_state_and_choice,
):
    (
        states_beginning_of_period,
        resources_beginning_of_period,
    ) = states_and_resources_beginning_of_period

    n_choices = len(choice_range)
    n_agents = len(resources_beginning_of_period)

    # Prepare random seeds for the simulation.
    key = jax.random.PRNGKey(basic_seed + period)
    sim_specific_keys = jax.random.split(key, num=n_agents + 2)

    # Interpolate policy and value function for all agents.
    policy, values_pre_taste_shock = interpolate_policy_and_value_for_all_agents(
        states_beginning_of_period=states_beginning_of_period,
        resources_beginning_of_period=resources_beginning_of_period,
        value_solved=value_solved,
        policy_left_solved=policy_left_solved,
        policy_right_solved=policy_right_solved,
        endog_grid_solved=endog_grid_solved,
        map_state_choice_to_index=map_state_choice_to_index,
        choice_range=choice_range,
        params=params,
        compute_utility=compute_utility,
    )

    # Draw taste shocks and calculate final value.
    taste_shocks = draw_taste_shocks(
        num_agents=n_agents,
        num_choices=n_choices,
        taste_shock_scale=params["lambda"],
        key=sim_specific_keys[0, :],
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
    savings_current_period = resources_beginning_of_period - consumption

    (
        resources_beginning_of_next_period,
        states_next_period,
        income_shocks_next_period,
    ) = transition_to_next_period(
        states_beginning_of_period=states_beginning_of_period,
        savings_current_period=savings_current_period,
        choice=choice,
        params=params,
        compute_exog_transition_vec=compute_exog_transition_vec,
        exog_state_mapping=exog_state_mapping,
        compute_beginning_of_period_resources=compute_beginning_of_period_resources,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
        sim_specific_keys=sim_specific_keys[1:, :],
    )
    carry = states_next_period, resources_beginning_of_next_period

    result = {
        "choice": choice,
        "consumption": consumption,
        "utility": utility_period,
        "taste_shocks": taste_shocks,
        "value": value_max,
        "savings": savings_current_period,
        "income_shock": income_shocks_next_period,
        **states_beginning_of_period,
    }

    return carry, result
