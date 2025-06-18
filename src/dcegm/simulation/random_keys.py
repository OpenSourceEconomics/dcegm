import jax
import jax.numpy as jnp


def draw_random_keys_for_seed(n_agents, n_periods, taste_shock_scale_is_scalar, seed):
    """Draw the random keys jax uses for a given seed."""

    # We start by determining the number of keys per period  for the three stochastic components
    # of the model. We will draw all keys together and assign them then based on index.
    # First: The transition of the exogenous processes. This happens vectorized over all
    # agents, therefore we need n_agents.
    n_stochastic_states_transition_keys = n_agents
    idx_1 = jnp.arange(n_stochastic_states_transition_keys)

    # Second is the income shock. So far, they are drawn for all agents at the same time,
    # therefore we have 1 key.
    n_keys_income_shock = 1
    idx_2 = n_stochastic_states_transition_keys

    # For the taste shock we need to distinguish if it is scalar or state specific
    if taste_shock_scale_is_scalar:
        n_keys_taste_shock_per_period = 1
        idx_3 = n_stochastic_states_transition_keys + 1
    else:
        n_keys_taste_shock_per_period = n_agents
        idx_3 = jnp.arange(
            start=n_stochastic_states_transition_keys + 1,
            stop=n_stochastic_states_transition_keys
            + n_keys_taste_shock_per_period
            + 1,
        )

    # Prepare random seeds for taste shocks
    n_keys_per_period = (
        n_stochastic_states_transition_keys
        + n_keys_taste_shock_per_period
        + n_keys_income_shock
    )
    sim_keys_draw = jnp.array(
        [
            jax.random.split(jax.random.PRNGKey(seed + period), num=n_keys_per_period)
            for period in range(n_periods)
        ]
    )

    sim_keys = {
        "stochastic_state_keys": sim_keys_draw[:-1, idx_1, :],
        "income_shock_keys": sim_keys_draw[:-1, idx_2, :],
        "taste_shock_keys": sim_keys_draw[:-1, idx_3, :],
    }

    # In the last period we only need the taste shock.
    last_period_sim_keys = {
        "taste_shock_keys": sim_keys_draw[-1, idx_3, :],
    }
    return sim_keys, last_period_sim_keys
