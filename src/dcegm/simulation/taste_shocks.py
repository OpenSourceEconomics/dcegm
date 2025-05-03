import jax
from jax import numpy as jnp


def draw_taste_shocks(
    params,
    states_beginning_of_period,
    n_choices,
    taste_shock_function,
    taste_shock_keys,
):
    n_agents = len(states_beginning_of_period["period"])
    if taste_shock_function["taste_shock_scale_is_scalar"]:
        taste_shock_scale = taste_shock_function["read_out_taste_shock_scale"](params)
        taste_shocks = draw_taste_shocks_scalar(
            n_agents, n_choices, taste_shock_scale, taste_shock_keys
        )
    else:
        taste_shocks = jax.vmap(
            draw_taste_shock_per_agent, in_axes=(0, 0, None, None, None)
        )(
            taste_shock_keys,
            states_beginning_of_period,
            params,
            n_choices,
            taste_shock_function,
        )

    return taste_shocks


def draw_taste_shock_per_agent(
    agent_key, agent_state, params, n_choices, taste_shock_function
):

    taste_shock_scale_agent = taste_shock_function["taste_shock_scale_per_state"](
        state_dict_vec=agent_state, params=params
    )

    taste_shocks = jax.random.gumbel(key=agent_key, shape=n_choices)

    taste_shocks = taste_shock_scale_agent * (taste_shocks - jnp.euler_gamma)
    return taste_shocks


def draw_taste_shocks_scalar(n_agents, n_choices, taste_shock_scale, key):
    taste_shocks = jax.random.gumbel(key=key, shape=(n_agents, n_choices))

    taste_shocks = taste_shock_scale * (taste_shocks - jnp.euler_gamma)
    return taste_shocks
