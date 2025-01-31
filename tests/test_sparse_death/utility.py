import jax
from jax import numpy as jnp


def create_utility_function_dict():
    return {
        "utility": utility,
        "marginal_utility": marginal_utility,
        "inverse_marginal_utility": inverse_marginal,
    }


def create_final_period_utility_function_dict():
    return {
        "utility": utiility_log_crra_final_consume_all,
        "marginal_utility": marginal_final,
    }


def utility(
    consumption: jnp.array,
    survival,
    choice,
    params,
):
    death = survival == 0
    utility_living = utility_alive(
        consumption=consumption,
        choice=choice,
        params=params,
    )
    utility_death = utiility_log_crra_final_consume_all(wealth=consumption)
    return death * utility_death + (1 - death) * utility_living


def utility_alive(
    consumption: jnp.array,
    choice,
    params,
):
    return jnp.log(consumption) - choice * params["delta"]


def marginal_utility(
    consumption,
    survival,
    choice,
    params,
):
    death = survival == 0

    marginal_alive = jax.jacfwd(
        lambda c: utility_alive(
            consumption=c,
            params=params,
            choice=choice,
        )
    )(consumption)

    marginal_death = marginal_final(consumption)

    return death * marginal_death + (1 - death) * marginal_alive


def inverse_marginal(
    marginal_utility,
):
    return 1 / marginal_utility


def utiility_log_crra_final_consume_all(
    wealth: jnp.array,
) -> jnp.array:
    return jnp.log(wealth)


def marginal_final(wealth):
    return 1 / wealth
