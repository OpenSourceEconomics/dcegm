import jax.numpy as jnp
from jax import vmap

from dcegm.law_of_motion import (
    calc_wealth_for_each_continuous_state_and_savings_grid_point,
    calc_wealth_for_each_savings_grid_point,
)


def adjust_observed_wealth(observed_states_dict, params, model):
    """Correct observed beginning of period wealth data for likelihood estimation.

    Wealth in empirical survey data is observed without the income of last period's
    choice.

    In the dcegm framework, however, individuals make their consumption decision given
    the income of the previous period. Therefore, agents in the model have a higher
    beginning of period wealth than the observed wealth in survey data.

    This function can be used to align these two wealth definitions; especially in
    likelihood estimation, where the computation of choice probabilities requires the
    correct beginning of period wealth.

    """
    observed_states_dict_int = observed_states_dict.copy()

    wealth_int = observed_states_dict["wealth"]
    savings_last_period = jnp.asarray(wealth_int / (1 + params["interest_rate"]))

    if len(model["options"]["exog_grids"]) == 2:
        # If there are two continuous states, we need to read out the second var
        second_cont_state_name = model["options"]["second_continuous_state_name"]
        second_cont_state_vars = observed_states_dict[second_cont_state_name]
        observed_states_dict_int.pop(second_cont_state_name)

        adjusted_wealth = vmap(
            calc_wealth_for_each_continuous_state_and_savings_grid_point,
            in_axes=(0, 0, 0, None, None, None, None),
        )(
            observed_states_dict_int,
            second_cont_state_vars,
            savings_last_period,
            jnp.array(0.0, dtype=jnp.float64),
            params,
            model["model_funcs"]["compute_assets_begin_of_period"],
            False,
        )

    else:
        adjusted_wealth = vmap(
            calc_wealth_for_each_savings_grid_point,
            in_axes=(0, 0, None, None, None, None),
        )(
            observed_states_dict,
            savings_last_period,
            jnp.array(0.0, dtype=jnp.float64),
            params,
            model["model_funcs"]["compute_assets_begin_of_period"],
            False,
        )

    return adjusted_wealth
