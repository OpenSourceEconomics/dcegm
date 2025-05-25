import jax.numpy as jnp
from jax import vmap

from dcegm.law_of_motion import (
    calc_assets_beginning_of_period_2cont_vec,
    calc_beginning_of_period_assets_1cont_vec,
)


def adjust_observed_assets(observed_states_dict, params, model_class):
    """Correct observed beginning of period assets data for likelihood estimation.

    Assets in empirical survey data is observed without the income of last period's
    choice.

    In the dcegm framework, however, individuals make their consumption decision given
    the income of the previous period. Therefore, agents in the model have higher
    beginning of assets wealth than the observed wealth in survey data.

    This function can be used to align these two wealth definitions; especially in
    likelihood estimation, where the computation of choice probabilities requires the
    correct beginning of period wealth.

    """
    observed_states_dict_int = observed_states_dict.copy()

    wealth_int = observed_states_dict["assets_begin_of_period"]
    interest_rate = model_class.model_funcs["read_funcs"]["interest_rate"](params)
    assets_end_last_period = jnp.asarray(wealth_int / (1 + interest_rate))

    model_funcs = model_class.model_funcs
    continuous_states_info = model_class.model_config["continuous_states_info"]

    if continuous_states_info["second_continuous_exists"]:
        # If there are two continuous states, we need to read out the second var
        second_cont_state_name = continuous_states_info["second_continuous_state_name"]
        second_cont_state_vars = observed_states_dict[second_cont_state_name]
        observed_states_dict_int.pop(second_cont_state_name)

        adjusted_assets = vmap(
            calc_assets_beginning_of_period_2cont_vec,
            in_axes=(0, 0, 0, None, None, None, None),
        )(
            observed_states_dict_int,
            second_cont_state_vars,
            assets_end_last_period,
            jnp.array(0.0, dtype=jnp.float64),
            params,
            model_funcs["compute_assets_begin_of_period"],
            False,
        )

    else:
        adjusted_assets = vmap(
            calc_beginning_of_period_assets_1cont_vec,
            in_axes=(0, 0, None, None, None, None),
        )(
            observed_states_dict,
            assets_end_last_period,
            jnp.array(0.0, dtype=jnp.float64),
            params,
            model_funcs["compute_assets_begin_of_period"],
            False,
        )

    return adjusted_assets
