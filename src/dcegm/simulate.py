from functools import partial

import jax
import jax.numpy as jnp
from dcegm.budget import calculate_resources_for_all_agents
from dcegm.egm.interpolate_marginal_utility import interpolate_policy_and_check_value
from dcegm.interpolation import get_index_high_and_low
from jax import vmap


def simulate_all_periods(
    states_period_0,
    wealth_period_0,
    num_periods,
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
    compute_beginning_of_period_wealth,
    exog_state_mapping,
    update_endog_state_by_state_and_choice,
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
        compute_beginning_of_period_wealth=compute_beginning_of_period_wealth,
        exog_state_mapping=exog_state_mapping,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    states_and_wealth_period_0 = states_period_0, wealth_period_0
    states_and_wealth_last_period, sim_data = jax.lax.scan(
        f=simulate_body, init=states_and_wealth_period_0, xs=jnp.arange(num_periods - 1)
    )
    # ToDo: Last period.

    return states_and_wealth_last_period, sim_data


def simulate_single_period(
    states_and_wealth_beginning_of_period,
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
    compute_beginning_of_period_wealth,
    exog_state_mapping,
    update_endog_state_by_state_and_choice,
):
    (
        states_beginning_of_period,
        wealth_beginning_of_period,
    ) = states_and_wealth_beginning_of_period

    # Read key variables of the simulation.
    num_choices = len(choice_range)

    # Prepare random seeds for the simulation.
    key = jax.random.PRNGKey(basic_seed + period)

    # Interpolate policy and value function for all agents.
    policy_agent, value_agent_pre_shock = interpolate_policy_and_value_for_all_agents(
        states_beginning_of_period=states_beginning_of_period,
        wealth_beginning_of_period=wealth_beginning_of_period,
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
    num_agents = wealth_beginning_of_period.shape[0]
    taste_shocks = draw_taste_shocks(
        num_agents=num_agents,
        num_choices=num_choices,
        taste_shock_scale=params["lambda"],
        key=key,
    )
    value_agent = value_agent_pre_shock + taste_shocks

    # Determine choice index of period by max value and select corresponding choice,
    # consumption and value.
    choice_index_period = jnp.nanargmax(value_agent, axis=1)
    choice_period = choice_range[choice_index_period]
    consumption_period = jnp.take_along_axis(
        policy_agent, choice_index_period[:, None], axis=1
    )[:, 0]
    value_period = jnp.take_along_axis(
        value_agent, choice_index_period[:, None], axis=1
    )[:, 0]

    # Calculate utility of period.
    utility_period = vmap(vectorized_utility, in_axes=(0, 0, 0, None, None))(
        consumption_period,
        states_beginning_of_period,
        choice_period,
        params,
        compute_utility,
    )
    # Calculate the savings.
    savings_this_period = wealth_beginning_of_period - consumption_period

    # Transition to next period.
    (
        wealth_at_beginning_of_next_period,
        states_next_period,
        income_shocks_next_period,
    ) = transition_to_next_period(
        states_beginning_of_period=states_beginning_of_period,
        savings_this_period=savings_this_period,
        choice_period=choice_period,
        params=params,
        compute_exog_transition_vec=compute_exog_transition_vec,
        exog_state_mapping=exog_state_mapping,
        compute_beginning_of_period_wealth=compute_beginning_of_period_wealth,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
        key=key,
    )
    carry = states_next_period, wealth_at_beginning_of_next_period

    result_data = {
        "choice": choice_period,
        "consumption": consumption_period,
        "utility": utility_period,
        "taste_shocks": taste_shocks,
        "value": value_period,
        "savings": savings_this_period,
        "income_shock": income_shocks_next_period,
        **states_beginning_of_period,
    }

    return carry, result_data


def interpolate_policy_and_value_for_all_agents(
    states_beginning_of_period,
    wealth_beginning_of_period,
    value_solved,
    policy_left_solved,
    policy_right_solved,
    endog_grid_solved,
    map_state_choice_to_index,
    choice_range,
    params,
    compute_utility,
):
    """This function interpolates the policy and value function for all agents.

    It uses the states at the beginning of period to select the solved policy and value
    and then interpolates the wealth at the beginning of period on them.

    """
    state_choice_indexes = get_state_choice_index_per_state(
        map_state_choice_to_index, states_beginning_of_period
    )
    value_grid_agent = jnp.take(value_solved, state_choice_indexes, axis=0)
    policy_left_grid_agent = jnp.take(policy_left_solved, state_choice_indexes, axis=0)
    policy_right_grid_agent = jnp.take(
        policy_right_solved, state_choice_indexes, axis=0
    )
    endog_grid_agent = jnp.take(endog_grid_solved, state_choice_indexes, axis=0)

    vectorized_interp = vmap(
        vmap(
            interpolate_policy_and_value_function_for_agents,
            in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )

    policy_agent, value_per_agent_interp = vectorized_interp(
        wealth_beginning_of_period,
        states_beginning_of_period,
        endog_grid_agent,
        value_grid_agent,
        policy_left_grid_agent,
        policy_right_grid_agent,
        choice_range,
        params,
        compute_utility,
    )
    return policy_agent, value_per_agent_interp


def transition_to_next_period(
    states_beginning_of_period,
    savings_this_period,
    choice_period,
    params,
    compute_exog_transition_vec,
    exog_state_mapping,
    compute_beginning_of_period_wealth,
    update_endog_state_by_state_and_choice,
    key,
):
    # Compute transitioning to next period. We need to do that separately for each
    # agent, therefore we need to split the random key.
    num_agents = savings_this_period.shape[0]
    sim_specific_keys = jax.random.split(key, num=num_agents)
    exog_states_next_period = vmap(
        realize_exog_process, in_axes=(0, 0, 0, None, None, None)
    )(
        states_beginning_of_period,
        choice_period,
        sim_specific_keys,
        params,
        compute_exog_transition_vec,
        exog_state_mapping,
    )

    endog_states_next_period = vmap(
        update_endog_for_one_agent, in_axes=(None, 0, 0, None)
    )(
        update_endog_state_by_state_and_choice,
        states_beginning_of_period,
        choice_period,
        params,
    )
    # Generate states next period and apply budged constraint for wealth at the
    # beginning of next period.
    # Initialize states by copying
    states_next_period = states_beginning_of_period.copy()
    # Then update
    states_to_update = {**endog_states_next_period, **exog_states_next_period}
    states_next_period.update(states_to_update)

    # Draw income shocks.
    income_shocks_next_period = draw_normal_shocks(
        key=key, num_agents=num_agents, mean=0, std=params["sigma"]
    )
    wealth_at_beginning_of_next_period = calculate_resources_for_all_agents(
        states_beginning_of_period=states_next_period,
        savings_end_of_last_period=savings_this_period,
        income_shocks_of_period=income_shocks_next_period,
        params=params,
        compute_beginning_of_period_wealth=compute_beginning_of_period_wealth,
    )
    return (
        wealth_at_beginning_of_next_period,
        states_next_period,
        income_shocks_next_period,
    )


def draw_normal_shocks(key, num_agents, mean=0, std=1):
    return jax.random.normal(key=key, shape=(num_agents,)) * std + mean


def update_endog_for_one_agent(update_func, state, choice, params):
    return update_func(params=params, **state, choice=choice)


def draw_taste_shocks(num_agents, num_choices, taste_shock_scale, key, mean=0):
    taste_shocks = jax.random.gumbel(key=key, shape=(num_agents, num_choices))
    taste_shocks = -jnp.euler_gamma + mean + taste_shock_scale * taste_shocks
    return taste_shocks


def vectorized_utility(consumption_period, state, choice, params, compute_utility):
    utility = compute_utility(
        consumption=consumption_period, params=params, choice=choice, **state
    )
    return utility


def realize_exog_process(state, choice, key, params, exog_func, exog_state_mapping):
    transition_vec = exog_func(params=params, **state, choice=choice)
    exog_proc_next_period = jax.random.choice(
        key=key, a=transition_vec.shape[0], p=transition_vec
    )
    exog_states_next_period = exog_state_mapping(exog_proc_next_period)
    return exog_states_next_period


def get_state_choice_index_per_state(map_state_choice_to_index, states):
    # select indexes by states
    indexes = map_state_choice_to_index[tuple((states[key],) for key in states.keys())]
    # As the code above generates a dummy dimension in the first we eliminate that
    return indexes[0]


def interpolate_policy_and_value_function_for_agents(
    wealth_beginning_of_period,
    state,
    endog_grid_agent,
    value_agent,
    policy_left_agent,
    policy_right_agent,
    choice,
    params,
    compute_utility,
):
    ind_high, ind_low = get_index_high_and_low(
        x=endog_grid_agent, x_new=wealth_beginning_of_period
    )
    state_choice_vec = {**state, "choice": choice}
    policy_interp, value_interp = interpolate_policy_and_check_value(
        policy_high=policy_left_agent[ind_high],
        value_high=value_agent[ind_high],
        wealth_high=endog_grid_agent[ind_high],
        policy_low=policy_right_agent[ind_low],
        value_low=value_agent[ind_low],
        wealth_low=endog_grid_agent[ind_low],
        new_wealth=wealth_beginning_of_period,
        compute_utility=compute_utility,
        endog_grid_min=endog_grid_agent[1],
        value_at_zero_wealth=value_agent[0],
        state_choice_vec=state_choice_vec,
        params=params,
    )
    return policy_interp, value_interp
