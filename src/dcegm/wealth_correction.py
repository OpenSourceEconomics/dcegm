import jax.numpy as jnp
from jax import vmap

from dcegm.law_of_motion import calc_wealth_for_each_savings_grid_point


def adjust_observed_wealth(observed_states_dict, wealth, params, model):
    """Correct wealth data, which is observed without the income of last period's
    choice.

    In the dcegm framework, individuals make their consumption decision given the income
    of the previous period. Therefore, agents in the model have a higher beginning of
    period wealth than the observed wealth in survey data. This function can be used to
    align these two wealth definitions.

    """
    savings_last_period = jnp.asarray(wealth / (1 + params["interest_rate"]))

    adjusted_resources = vmap(
        calc_wealth_for_each_savings_grid_point, in_axes=(0, 0, None, None, None)
    )(
        observed_states_dict,
        savings_last_period,
        jnp.array(0.0, dtype=jnp.float64),
        params,
        model["model_funcs"]["compute_beginning_of_period_resources"],
    )

    return adjusted_resources
