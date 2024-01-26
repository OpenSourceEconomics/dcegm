from typing import Any
from typing import Callable
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from dcegm.egm.aggregate_marginal_utility import (
    calculate_choice_probs_and_unsqueezed_logsum,
)
from dcegm.egm.interpolate_marginal_utility import (
    interp_value_and_check_creditconstraint,
)
from dcegm.interpolation import get_index_high_and_low
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.params import process_params
from dcegm.pre_processing.setup_model import setup_model
from dcegm.simulation.sim_utils import get_state_choice_index_per_state
from dcegm.solve import backward_induction


def create_individual_likelihood_function(
    observed_states: Dict[str, int],
    observed_wealth: np.array,
    observed_choices: np.array,
    options: Dict[str, Any],
    exog_savings_grid: np.ndarray,
    state_space_functions: Dict[str, Callable],
    utility_functions: Dict[str, Callable],
    budget_constraint: Callable,
    utility_functions_final_period: Dict[str, Callable],
    params_all=None,
):
    model = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        options["model_params"]["quadrature_points_stochastic"]
    )

    if params_all is None:
        # If params_all is not supplied, all elements of params will be estimated and
        # we use the params supplied at each iteration for the solution of the model.
        def update_params(params):
            params_initial = process_params(params)
            return params_initial

    else:
        # If params_all is supplied, we use the params supplied at each iteration for
        # updating params_all and then used the updated params_all for the solution of
        # the model.
        def update_params(params):
            params_update = params_all.copy()
            params_update.update(params)
            params_initial = process_params(params_update)
            return params_initial

    # Create a solution function, which only takes the parameters as input.
    def partial_backwards_induction(params_in):
        return backward_induction(
            params=params_in,
            period_specific_state_objects=model["period_specific_state_objects"],
            exog_savings_grid=exog_savings_grid,
            state_space=model["state_space"],
            income_shock_draws_unscaled=income_shock_draws_unscaled,
            income_shock_weights=income_shock_weights,
            n_periods=options["state_space"]["n_periods"],
            model_funcs=model["model_funcs"],
            compute_upper_envelope=model["compute_upper_envelope"],
        )

    observed_state_choice_indexes = create_observed_choice_indexes(
        observed_states_dict=observed_states,
        model=model,
    )

    # Create the calculation of the choice probabilities, which takes parameters as
    # input as well as the solved endogenous wealth grid and the values.
    def partial_choice_prob_calculation(value_in, endog_grid_in, params_in):
        return calc_observed_choice_probabilities(
            value_solved=value_in,
            endog_grid_solved=endog_grid_in,
            params=params_in,
            observed_states=observed_states,
            observed_choices=observed_choices,
            state_choice_indexes=observed_state_choice_indexes,
            oberseved_wealth=observed_wealth,
            choice_range=options["model_params"]["choice_range"],
            compute_utility=model["model_funcs"]["utility"],
        )

    def individual_likelihood(params):
        params_initial = update_params(params)
        (
            value_solved,
            policy_left_solved,
            policy_right_solved,
            endog_grid_solved,
        ) = partial_backwards_induction(params_initial)
        choice_probs = partial_choice_prob_calculation(
            value_in=value_solved,
            endog_grid_in=endog_grid_solved,
            params_in=params_initial,
        )
        return choice_probs

    return jax.jit(individual_likelihood)


def calc_observed_choice_probabilities(
    value_solved,
    endog_grid_solved,
    params,
    observed_states,
    observed_choices,
    state_choice_indexes,
    oberseved_wealth,
    choice_range,
    compute_utility,
):
    """This function interpolates the policy and value function for all agents.

    It uses the states at the beginning of period to select the solved policy and value
    and then interpolates the wealth at the beginning of period on them.

    """

    value_grid_agent = jnp.take(
        value_solved, state_choice_indexes, axis=0, mode="fill", fill_value=jnp.nan
    )
    endog_grid_agent = jnp.take(endog_grid_solved, state_choice_indexes, axis=0)
    vectorized_interp = jax.vmap(
        jax.vmap(
            interpolate_value_and_calc_choice_probabilities,
            in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )

    value_per_agent_interp = vectorized_interp(
        observed_states,
        oberseved_wealth,
        endog_grid_agent,
        value_grid_agent,
        choice_range,
        params,
        compute_utility,
    )
    choice_prob_across_choices, _, _ = calculate_choice_probs_and_unsqueezed_logsum(
        choice_values_per_state=value_per_agent_interp,
        taste_shock_scale=params["lambda"],
    )
    choice_probs = jnp.take_along_axis(
        choice_prob_across_choices, observed_choices[:, None], axis=1
    )[:, 0]
    return choice_probs


def interpolate_value_and_calc_choice_probabilities(
    resources_beginning_of_period,
    state,
    endog_grid_agent,
    value_agent,
    choice,
    params,
    compute_utility,
):
    ind_high, ind_low = get_index_high_and_low(
        x=endog_grid_agent, x_new=resources_beginning_of_period
    )
    state_choice_vec = {**state, "choice": choice}
    policy_interp, value_interp = interp_value_and_check_creditconstraint(
        value_high=value_agent[ind_high],
        wealth_high=endog_grid_agent[ind_high],
        value_low=value_agent[ind_low],
        wealth_low=endog_grid_agent[ind_low],
        new_wealth=resources_beginning_of_period,
        compute_utility=compute_utility,
        endog_grid_min=endog_grid_agent[1],
        value_at_zero_wealth=value_agent[0],
        state_choice_vec=state_choice_vec,
        params=params,
    )

    return value_interp


def create_observed_choice_indexes(
    observed_states_dict: Dict[str, int],
    model: [Dict, Any],
):
    observed_state_choice_indexes = get_state_choice_index_per_state(
        map_state_choice_to_index=model["map_state_choice_to_index"],
        states=observed_states_dict,
        state_space_names=model["state_space_names"],
    )
    return observed_state_choice_indexes
