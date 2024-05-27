from typing import Any
from typing import Dict

import jax.numpy as jnp


def create_utility_function_dict():
    """Create dictionary with utility functions.

    Returns:
        utility_functions (dict): Dictionary with utility functions.

    """
    return {
        "utility": utility_crra,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
    }


def utiility_log_crra(
    consumption: jnp.array,
    choice: int,
    params: Dict[str, float],  # delta: float
) -> jnp.array:
    """Compute the agent's utility in case of theta equal to 1.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (jnp.array): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    return jnp.log(consumption) - (1 - choice) * params["delta"]


def utiility_log_crra_final_consume_all(
    resources: jnp.array,
    choice: int,
    params: Dict[str, float],  # delta: float
) -> jnp.array:
    """Compute the agent's utility in case of theta equal to 1.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (jnp.array): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    return jnp.log(resources) - (1 - choice) * params["delta"]


def utility_crra(
    consumption: jnp.array,
    choice: int,
    params: Dict[str, float],
) -> jnp.array:
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (jnp.array): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """

    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])

    utility = utility_consumption - (1 - choice) * params["delta"]

    return utility


def marginal_utility_crra(
    consumption: jnp.array, params: Dict[str, float], choice, options: Dict[str, Any]
) -> jnp.array:
    """Computes marginal utility of CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        marginal_utility (jnp.array): Marginal utility of CRRA consumption
            function. Array of shape (n_quad_stochastic * n_grid_wealth,).

    """
    marginal_utility = consumption ** (-params["rho"])

    return marginal_utility


def inverse_marginal_utility_crra(
    marginal_utility: jnp.array,
    params: Dict[str, float],
) -> jnp.array:
    """Computes the inverse marginal utility of a CRRA utility function.

    Args:
        marginal_utility (jnp.array): Level of marginal CRRA utility.
            Array of shape (n_grid_wealth,).
        params (dict): Dictionary containing model parameters.

    Returns:
        inverse_marginal_utility(jnp.array): Inverse of the marginal utility of
            a CRRA consumption function. Array of shape (n_grid_wealth,).

    """
    inverse_marginal_utility = marginal_utility ** (-1 / params["rho"])

    return inverse_marginal_utility


def create_final_period_utility_function_dict():
    """Create dictionary with utility functions for the final period.

    Returns:
        utility_functions_final_period (dict): Dictionary with utility functions
            for the final period.

    """
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


def utility_final_consume_all(
    choice: int,
    resources: jnp.array,
    params: Dict[str, float],
):
    util_consumption = (resources ** (1 - params["rho"]) - 1) / (1 - params["rho"])
    util = util_consumption - (1 - choice) * params["delta"]

    return util


def marginal_utility_final_consume_all(
    choice, resources: jnp.array, params: Dict[str, float], options: Dict[str, Any]
) -> jnp.array:
    """Computes marginal utility of CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        marginal_utility (jnp.array): Marginal utility of CRRA consumption
            function. Array of shape (n_quad_stochastic * n_grid_wealth,).

    """
    marg_util = resources ** (-params["rho"])

    return marg_util
