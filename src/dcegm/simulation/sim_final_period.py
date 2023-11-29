import jax
import numpy as np
from dcegm.simulation.sim_utils import compute_final_utility_for_each_choice
from dcegm.simulation.sim_utils import draw_taste_shocks
from dcegm.simulation.sim_utils import get_state_choice_index_per_state
from jax import numpy as jnp
from jax import vmap


def simulate_final_period(
    states_and_resources_beginning_of_period,
    period,
    params,
    basic_seed,
    choice_range,
    map_state_choice_to_index,
    compute_utility_final_period,
):
    (
        states_beginning_of_final_period,
        resources_beginning_of_final_period,
    ) = states_and_resources_beginning_of_period

    n_choices = len(choice_range)
    n_agents = len(resources_beginning_of_final_period)

    utilities_pre_taste_shock = vmap(
        vmap(
            compute_final_utility_for_each_choice,
            in_axes=(None, 0, None, None, None),
        ),
        in_axes=(0, None, 0, None, None),
    )(
        states_beginning_of_final_period,
        choice_range,
        resources_beginning_of_final_period,
        params,
        compute_utility_final_period,
    )
    state_choice_indexes = get_state_choice_index_per_state(
        map_state_choice_to_index, states_beginning_of_final_period
    )
    utilities_pre_taste_shock = jnp.where(
        state_choice_indexes < 0, np.nan, utilities_pre_taste_shock
    )

    # Draw taste shocks and calculate final value.
    key = jax.random.PRNGKey(basic_seed + period)
    taste_shocks = draw_taste_shocks(
        num_agents=n_agents,
        num_choices=n_choices,
        taste_shock_scale=params["lambda"],
        key=key,
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
        "consumption": resources_beginning_of_final_period,
        "utility": utility_period,
        "value": value_period,
        "taste_shocks": taste_shocks[np.newaxis, :, :],
        "savings": np.zeros_like(utility_period),
        "income_shock": np.zeros(n_agents),
        **states_beginning_of_final_period,
    }

    return result
