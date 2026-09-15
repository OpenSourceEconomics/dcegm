import jax
import jax.numpy as jnp
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
        realize_stochastic_states, in_axes=(0, 0, 0, 0, None, None)
    )(
        discrete_states_beginning_of_period,
        continuous_state_beginning_of_period,
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

    # Neither beginning-of-period wealth nor the additional continuous states are
    # computed here. Both are applied at the start of the period they belong to
    # (see assets_begin_of_period_for_each_choice and
    # continuous_state_begin_of_period_for_each_choice), because either law-of-motion
    # function may depend on that period's own choice -- which is not known yet at
    # this point.
    return discrete_states_next_period, income_shocks_next_period


def continuous_state_begin_of_period_for_each_choice(
    discrete_states_beginning_of_period,
    continuous_state_of_previous_period,
    choice_range,
    params,
    compute_continuous_state,
    continuous_state_depends_on_choice,
):
    """Additional continuous states at the start of a period, for every choice.

    The mirror image of ``assets_begin_of_period_for_each_choice`` below, for the
    *other* law-of-motion function. When ``next_period_continuous_state`` declares
    ``choice``, this period's continuous state differs by the choice about to be made,
    so it -- like wealth -- must be evaluated for every choice before the choice is
    drawn, and the agent then compares each choice's value at that choice's own
    continuous state.

    Returns a dict of arrays of shape ``(n_agents, n_choices)``. When the transition
    does not declare ``choice`` the single result is repeated across the choice axis,
    keeping one uniform shape downstream at no extra cost.

    """
    n_agents = next(iter(discrete_states_beginning_of_period.values())).shape[0]
    n_choices = len(choice_range)

    def _for_choice(choice_value):
        states = discrete_states_beginning_of_period
        if continuous_state_depends_on_choice:
            states = {
                **states,
                "choice": jnp.full(n_agents, choice_value, dtype=choice_range.dtype),
            }
        return calculate_second_continuous_state_for_all_agents(
            discrete_states_beginning_of_period=states,
            continuous_state_beginning_of_period=continuous_state_of_previous_period,
            params=params,
            compute_continuous_state=compute_continuous_state,
        )

    if not continuous_state_depends_on_choice:
        per_agent = _for_choice(choice_range[0])
        return {
            key: jnp.repeat(val[:, None], n_choices, axis=1)
            for key, val in per_agent.items()
        }

    # (n_choices, n_agents) -> (n_agents, n_choices)
    per_choice = vmap(_for_choice)(choice_range)
    return {key: val.T for key, val in per_choice.items()}


def assets_begin_of_period_for_each_choice(
    states_beginning_of_period,
    continuous_state_per_choice,
    assets_end_of_previous_period,
    income_shock,
    choice_range,
    params,
    compute_assets_begin_of_period,
    budget_depends_on_choice,
):
    """Beginning-of-period wealth for every choice, before any choice is made.

    Applied at the *start* of a period, using the state that period was entered with,
    last period's end-of-period assets, and this period's income shock.

    Returns arrays of shape ``(n_agents, n_choices)``. When the budget equation declares
    ``choice`` the agent faces a genuinely different wealth per choice (e.g. a choice-
    specific cost), and the discrete choice is then made by comparing each choice's
    value *at its own wealth* -- which is exactly what the solved solution is indexed
    by. When it does not, the single wealth is repeated across the choice axis so
    everything downstream keeps one uniform shape, without paying for redundant budget
    evaluations.

    """
    n_choices = len(choice_range)

    def _for_choice(choice_index, choice_value):
        states = states_beginning_of_period
        if continuous_state_per_choice is not None:
            # The budget reads the continuous state, which is itself per choice.
            states = {
                **states,
                **{
                    name: val[:, choice_index]
                    for name, val in continuous_state_per_choice.items()
                },
            }
        if budget_depends_on_choice:
            states = {
                **states,
                "choice": jnp.full(
                    assets_end_of_previous_period.shape[0],
                    choice_value,
                    dtype=choice_range.dtype,
                ),
            }
        return calculate_assets_begin_of_period_for_all_agents(
            states_beginning_of_period=states,
            asset_grid_point_end_of_previous_period=assets_end_of_previous_period,
            income_shocks_of_period=income_shock,
            params=params,
            compute_assets_begin_of_period=compute_assets_begin_of_period,
        )

    continuous_state_is_per_choice = continuous_state_per_choice is not None and any(
        val.shape[1] > 1 for val in continuous_state_per_choice.values()
    )
    if not budget_depends_on_choice and not continuous_state_is_per_choice:
        assets, aux = _for_choice(0, choice_range[0])
        assets = jnp.repeat(assets[:, None], n_choices, axis=1)
        aux = {
            key: jnp.repeat(val[:, None], n_choices, axis=1) for key, val in aux.items()
        }
        return assets, aux

    # Python loop over the (small, static) choice axis rather than vmap: the
    # per-choice continuous state is sliced by a concrete index.
    per_choice = [
        _for_choice(idx, choice_value) for idx, choice_value in enumerate(choice_range)
    ]
    assets = jnp.stack([a for a, _ in per_choice], axis=1)
    aux_keys = per_choice[0][1].keys()
    aux = {
        key: jnp.stack([aux_j[key] for _, aux_j in per_choice], axis=1)
        for key in aux_keys
    }
    return assets, aux


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


def draw_normal_shocks(key, num_agents, mean, std):
    return jax.random.normal(key=key, shape=(num_agents,)) * std + mean


def update_discrete_states_for_one_agent(update_func, state, choice, params):
    return update_func(**state, choice=choice, params=params)


def vectorized_utility(consumption_period, state, choice, params, compute_utility):
    utility = compute_utility(
        consumption=consumption_period, params=params, choice=choice, **state
    )
    return utility


def realize_stochastic_states(
    state, cont_state, choice, key, params, processed_stochastic_funcs
):
    if cont_state is not None:
        all_states = {**state, **cont_state}
    else:
        all_states = state
    stochastic_states_next_period = {}
    for state_name in processed_stochastic_funcs.keys():
        key, subkey = jax.random.split(key)
        state_vec = processed_stochastic_funcs[state_name](
            params=params, **all_states, choice=choice
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
