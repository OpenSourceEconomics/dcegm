import jax
import numpy as np
import pandas as pd
from dcegm.budget import calculate_resources_for_all_agents
from dcegm.egm.interpolate_marginal_utility import interpolate_policy_and_check_value
from dcegm.interpolation import get_index_high_and_low
from jax import numpy as jnp
from jax import vmap


def interpolate_policy_and_value_for_all_agents(
    states_beginning_of_period,
    resources_beginning_of_period,
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

    value_grid_agent = jnp.take(
        value_solved, state_choice_indexes, axis=0, mode="fill", fill_value=jnp.nan
    )
    policy_left_grid_agent = jnp.take(policy_left_solved, state_choice_indexes, axis=0)
    policy_right_grid_agent = jnp.take(
        policy_right_solved, state_choice_indexes, axis=0
    )
    endog_grid_agent = jnp.take(endog_grid_solved, state_choice_indexes, axis=0)

    vectorized_interp = vmap(
        vmap(
            interpolate_policy_and_value_function,
            in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )

    policy_agent, value_per_agent_interp = vectorized_interp(
        resources_beginning_of_period,
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
    savings_current_period,
    choice,
    params,
    compute_exog_transition_vec,
    exog_state_mapping,
    compute_beginning_of_period_resources,
    update_endog_state_by_state_and_choice,
    sim_specific_keys,
):
    n_agents = savings_current_period.shape[0]
    exog_states_next_period = vmap(
        realize_exog_process, in_axes=(0, 0, 0, None, None, None)
    )(
        states_beginning_of_period,
        choice,
        sim_specific_keys[1:, :],
        params,
        compute_exog_transition_vec,
        exog_state_mapping,
    )

    endog_states_next_period = vmap(
        update_endog_for_one_agent, in_axes=(None, 0, 0, None)
    )(
        update_endog_state_by_state_and_choice,
        states_beginning_of_period,
        choice,
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
        key=sim_specific_keys[0, :], num_agents=n_agents, mean=0, std=params["sigma"]
    )
    resources_beginning_of_next_period = calculate_resources_for_all_agents(
        states_beginning_of_period=states_next_period,
        savings_end_of_previous_period=savings_current_period,
        income_shocks_of_period=income_shocks_next_period,
        params=params,
        compute_beginning_of_period_resources=compute_beginning_of_period_resources,
    )
    return (
        resources_beginning_of_next_period,
        states_next_period,
        income_shocks_next_period,
    )


def compute_final_utility_for_each_choice(
    state_vec, choice, resources, params, compute_utility_final_period
):
    util = compute_utility_final_period(
        **state_vec,
        choice=choice,
        resources=resources,
        params=params,
    )

    return util


def draw_normal_shocks(key, num_agents, mean=0, std=1):
    return jax.random.normal(key=key, shape=(num_agents,)) * std + mean


def update_endog_for_one_agent(update_func, state, choice, params):
    return update_func(params=params, **state, choice=choice)


def draw_taste_shocks(n_agents, n_choices, taste_shock_scale, key):
    taste_shocks = jax.random.gumbel(key=key, shape=(n_agents, n_choices))
    taste_shocks = taste_shock_scale * (taste_shocks - jnp.euler_gamma)
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


def interpolate_policy_and_value_function(
    resources_beginning_of_period,
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
        x=endog_grid_agent, x_new=resources_beginning_of_period
    )
    state_choice_vec = {**state, "choice": choice}
    policy_interp, value_interp = interpolate_policy_and_check_value(
        policy_high=policy_left_agent[ind_high],
        value_high=value_agent[ind_high],
        wealth_high=endog_grid_agent[ind_high],
        policy_low=policy_right_agent[ind_low],
        value_low=value_agent[ind_low],
        wealth_low=endog_grid_agent[ind_low],
        new_wealth=resources_beginning_of_period,
        compute_utility=compute_utility,
        endog_grid_min=endog_grid_agent[1],
        value_at_zero_wealth=value_agent[0],
        state_choice_vec=state_choice_vec,
        params=params,
    )
    return policy_interp, value_interp


def create_simulation_df(sim_dict):
    n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

    keys_to_drop = ["taste_shocks", "period"]
    dict_to_df = {key: sim_dict[key] for key in sim_dict if key not in keys_to_drop}

    df_without_taste_shocks = pd.DataFrame(
        {key: val.ravel() for key, val in dict_to_df.items()},
        index=pd.MultiIndex.from_product(
            [np.arange(n_periods), np.arange(n_agents)],
            names=["period", "agent"],
        ),
    )

    taste_shocks = sim_dict["taste_shocks"]
    df_taste_shocks = pd.DataFrame(
        {
            f"taste_shock_{choice}": taste_shocks[..., choice].flatten()
            for choice in range(n_choices)
        },
        index=pd.MultiIndex.from_product(
            [np.arange(n_periods), np.arange(n_agents)],
            names=["period", "agent"],
        ),
    )

    df_combined = df_without_taste_shocks.join(df_taste_shocks)

    return df_combined
