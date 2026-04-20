import jax
import numpy as np
import pandas as pd
from jax import vmap

from dcegm.law_of_motion import (
    calculate_assets_begin_of_period_for_all_agents,
    calculate_second_continuous_state_for_all_agents,
)


def transition_to_next_period(
    discrete_states_beginning_of_period,
    continuous_state_beginning_of_period,
    assets_end_of_period,
    choice,
    params,
    model_funcs_sim,
    read_funcs,
    sim_keys,
):
    n_agents = assets_end_of_period.shape[0]

    stochastic_states_next_period = vmap(
        realize_stochastic_states, in_axes=(0, 0, 0, None, None)
    )(
        discrete_states_beginning_of_period,
        choice,
        sim_keys["stochastic_state_keys"],
        params,
        model_funcs_sim["processed_stochastic_funcs"],
    )

    discrete_endog_states_next_period = vmap(
        update_discrete_states_for_one_agent, in_axes=(None, 0, 0, None)  # choice
    )(
        model_funcs_sim["next_period_deterministic_state"],
        discrete_states_beginning_of_period,
        choice,
        params,
    )

    # Generate states next period and apply budged constraint for wealth at the
    # beginning of next period.
    # Initialize states by copying
    discrete_states_next_period = discrete_states_beginning_of_period.copy()
    states_to_update = {
        **discrete_endog_states_next_period,
        **stochastic_states_next_period,
    }
    discrete_states_next_period.update(states_to_update)

    # Overwrite datatype with discrete_states_beginning_of_period dtypes
    for key in discrete_states_next_period.keys():
        discrete_states_next_period[key] = discrete_states_next_period[key].astype(
            discrete_states_beginning_of_period[key].dtype
        )

    income_shock_std = read_funcs["income_shock_std"](params)
    income_shock_mean = read_funcs["income_shock_mean"](params)

    # Draw income shocks.
    income_shocks_next_period = draw_normal_shocks(
        key=sim_keys["income_shock_keys"],
        num_agents=n_agents,
        mean=income_shock_mean,
        std=income_shock_std,
    )

    next_period_wealth = model_funcs_sim["compute_assets_begin_of_period"]
    if continuous_state_beginning_of_period is not None:

        continuous_state_next_period = calculate_second_continuous_state_for_all_agents(
            discrete_states_beginning_of_period=discrete_states_next_period,
            continuous_state_beginning_of_period=continuous_state_beginning_of_period,
            params=params,
            compute_continuous_state=model_funcs_sim["next_period_continuous_state"],
        )

        all_states_next_period = {
            **discrete_states_next_period,
            **continuous_state_next_period,
        }
    else:
        all_states_next_period = discrete_states_next_period.copy()
        continuous_state_next_period = None

    assets_beginning_of_next_period, budget_aux = (
        calculate_assets_begin_of_period_for_all_agents(
            states_beginning_of_period=all_states_next_period,
            asset_grid_point_end_of_previous_period=assets_end_of_period,
            income_shocks_of_period=income_shocks_next_period,
            params=params,
            compute_assets_begin_of_period=next_period_wealth,
        )
    )

    return (
        assets_beginning_of_next_period,
        budget_aux,
        discrete_states_next_period,
        continuous_state_next_period,
        income_shocks_next_period,
    )


def compute_final_utility_for_each_choice(
    state_vec, choice, wealth, params, compute_utility_final_period
):
    util = compute_utility_final_period(
        **state_vec,
        choice=choice,
        wealth=wealth,
        params=params,
    )

    return util


def draw_normal_shocks(key, num_agents, mean=0, std=1):
    return jax.random.normal(key=key, shape=(num_agents,)) * std + mean


def update_discrete_states_for_one_agent(update_func, state, choice, params):
    return update_func(**state, choice=choice, params=params)


def next_period_continuous_state_for_one_agent(
    update_func, discrete_states, continuous_state, choice, params
):

    return update_func(
        **discrete_states,
        continuous_state=continuous_state,
        choice=choice,
        params=params,
    )


def vectorized_utility(consumption_period, state, choice, params, compute_utility):
    utility = compute_utility(
        consumption=consumption_period, params=params, choice=choice, **state
    )
    return utility


def realize_stochastic_states(state, choice, key, params, processed_stochastic_funcs):
    stochastic_states_next_period = {}
    for state_name in processed_stochastic_funcs.keys():
        key, subkey = jax.random.split(key)
        state_vec = processed_stochastic_funcs[state_name](
            params=params, **state, choice=choice
        )
        stochastic_states_next_period[state_name] = jax.random.choice(
            key=subkey, a=state_vec.shape[0], p=state_vec
        )
    return stochastic_states_next_period


def create_simulation_df(sim_dict):
    n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

    keys_to_drop = ["taste_shocks", "period", "value_choice"]
    dict_to_df = {key: sim_dict[key] for key in sim_dict if key not in keys_to_drop}

    df = pd.DataFrame(
        {key: val.ravel() for key, val in dict_to_df.items()},
        index=pd.MultiIndex.from_product(
            [np.arange(n_periods), np.arange(n_agents)],
            names=["period", "agent"],
        ),
    )

    for choice_var in ["taste_shocks", "value_choice"]:
        df_choice = pd.DataFrame(
            {
                f"{choice_var}_{choice}": sim_dict[choice_var][..., choice].flatten()
                for choice in range(n_choices)
            },
            index=pd.MultiIndex.from_product(
                [np.arange(n_periods), np.arange(n_agents)],
                names=["period", "agent"],
            ),
        )

        df = df.join(df_choice)

    return df
