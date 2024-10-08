import jax.numpy as jnp
from jax import vmap

from dcegm.law_of_motion import (
    calc_wealth_for_each_continuous_state_and_savings_grid_point,
    calc_wealth_for_each_savings_grid_point,
)


def adjust_observed_wealth(observed_states_dict, params, model):
    """Correct wealth data, which is observed without the income of last period's
    choice.

    In the dcegm framework, individuals make their consumption decision given the income
    of the previous period. Therefore, agents in the model have a higher beginning of
    period wealth than the observed wealth in survey data. This function can be used to
    align these two wealth definitions.

    """
    observed_states_dict_int = observed_states_dict.copy()

    wealth_int = observed_states_dict["wealth"]
    observed_states_dict.pop("wealth")
    savings_last_period = jnp.asarray(wealth_int / (1 + params["interest_rate"]))

    if len(model["options"]["exog_grids"]) == 2:
        # If there are two continuous states, we need to read out the second var
        second_cont_state_name = model["options"]["second_continuous_state_name"]
        second_cont_state_vars = observed_states_dict[second_cont_state_name]
        observed_states_dict_int.pop(second_cont_state_name)

        adjusted_wealth = vmap(
            calc_wealth_for_each_continuous_state_and_savings_grid_point,
            in_axes=(0, 0, 0, None, None, None),
        )(
            observed_states_dict_int,
            second_cont_state_vars,
            savings_last_period,
            jnp.array(0.0, dtype=jnp.float64),
            params,
            model["model_funcs"]["compute_beginning_of_period_wealth"],
        )

    else:
        adjusted_wealth = vmap(
            calc_wealth_for_each_savings_grid_point, in_axes=(0, 0, None, None, None)
        )(
            observed_states_dict,
            savings_last_period,
            jnp.array(0.0, dtype=jnp.float64),
            params,
            model["model_funcs"]["compute_beginning_of_period_wealth"],
        )

    return adjusted_wealth
