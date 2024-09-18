import jax.numpy as jnp
from jax import vmap

from dcegm.budget import calculate_resources_for_each_grid_point


def adjust_observed_wealth(observed_states_dict, wealth, params, model):
    """This function corrects the wealth data, which is observed without the income of
    last periods choice.

    In the dcegm framework, individuals make their consumption decision given the income
    of last period. Therefore agents in the model have a higher beginning of period
    wealth than the observed wealth in survey data. This function can be used to align
    this two deifinitions of wealth.

    """
    savings_last_period = jnp.asarray(wealth / (1 + params["interest_rate"]))

    adjusted_resources = vmap(
        calculate_resources_for_each_grid_point, in_axes=(0, 0, None, None, None)
    )(
        observed_states_dict,
        savings_last_period,
        jnp.array(0.0, dtype=jnp.float64),
        params,
        model["model_funcs"]["compute_beginning_of_period_resources"],
    )

    return adjusted_resources
