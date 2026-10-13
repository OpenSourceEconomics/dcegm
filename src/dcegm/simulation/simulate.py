"""The simulation function."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from dcegm.interfaces.index_functions import get_state_choice_index_per_discrete_states
from dcegm.interpolation.simulation_interp import (
    interpolate_policy_and_value_for_all_agents,
)
from dcegm.simulation.random_keys import draw_random_keys_for_seed
from dcegm.simulation.sim_utils import (
    assets_begin_of_period_for_each_choice,
    compute_final_utility_for_each_choice,
    draw_income_shocks_for_period,
    transition_to_next_period,
    vectorized_utility,
)
from dcegm.simulation.taste_shocks import draw_taste_shocks


def _check_initial_assets(states_initial):
    """Reject the pre-``assets_end_of_previous_period`` initial-state contract.

    Passing ``assets_begin_of_period`` used to mean "period 0's wealth, take it as
    given", which made period 0 the one period whose budget equation never ran. Now
    every period builds its own wealth, so what the first period needs is what it
    inherits, not what it ends up with -- and silently reading the old key would quietly
    reinterpret its value.

    """
    if "assets_end_of_previous_period" in states_initial:
        return
    if "assets_begin_of_period" in states_initial:
        raise ValueError(
            "states_initial must contain 'assets_end_of_previous_period', not "
            "'assets_begin_of_period'. Every period, the first included, computes "
            "its beginning-of-period wealth from the budget equation, so supply "
            "the assets agents carry into the first period; its wealth then "
            "follows from the budget equation as in every other period."
        )
    raise ValueError(
        "states_initial must contain 'assets_end_of_previous_period': the assets "
        "agents carry into the first period, from which the budget equation builds "
        "that period's wealth."
    )


def simulate_all_periods(
    states_initial,
    n_periods,
    params,
    seed,
    endog_grid_solved,
    policy_solved,
    value_solved,
    model_structure,
    model_funcs,
    model_config,
    alt_model_funcs_sim,
):
    alt_model_funcs_sim = (
        model_funcs if alt_model_funcs_sim is None else alt_model_funcs_sim
    )

    discrete_state_space = model_structure["state_space_dict"]

    # Set initial states to internal dtype
    states_initial_dtype = {
        key: value.astype(discrete_state_space[key].dtype)
        for key, value in states_initial.items()
        if key in discrete_state_space
    }
    # Every period, the first included, builds its own beginning-of-period wealth
    # from the budget equation (see simulate_single_period), so the carry holds what
    # that needs -- last period's end-of-period assets -- rather than a ready-made
    # wealth level. That is why the user supplies assets_end_of_previous_period:
    # agents enter period 0 out of a notional period -1 whose savings they carry, and
    # period 0's wealth follows from the budget equation like any other period's.
    _check_initial_assets(states_initial)
    states_initial_dtype["assets_end_of_previous_period"] = states_initial[
        "assets_end_of_previous_period"
    ]

    if "dummy_stochastic" in model_structure["stochastic_states_names"]:
        states_initial_dtype["dummy_stochastic"] = jnp.zeros_like(
            states_initial_dtype["period"]
        )

    continuous_states_info = model_config["continuous_states_info"]
    has_additional_continuous_state = continuous_states_info[
        "has_additional_continuous_state"
    ]
    additional_continuous_state_names = continuous_states_info[
        "additional_continuous_state_names"
    ]

    if has_additional_continuous_state:
        for name in additional_continuous_state_names:
            # Carried as this period's realised value: unlike wealth, it is settled
            # at the end of the previous period, so the user's array is simply the
            # first period's own state.
            states_initial_dtype[name] = states_initial[name]

    n_agents = len(states_initial["period"])

    # Draw the random keys
    sim_keys, last_period_sim_keys = draw_random_keys_for_seed(
        n_agents=n_agents,
        n_periods=n_periods,
        taste_shock_scale_is_scalar=alt_model_funcs_sim["taste_shock_function"][
            "taste_shock_scale_is_scalar"
        ],
        seed=seed,
    )
    read_funcs = model_funcs["read_funcs"]

    simulate_body = partial(
        simulate_single_period,
        params=params,
        endog_grid_solved=endog_grid_solved,
        value_solved=value_solved,
        policy_solved=policy_solved,
        model_structure_sol=model_structure,
        model_funcs_sim=alt_model_funcs_sim,
        compute_utility=model_funcs["compute_utility"],
        read_funcs=read_funcs,
        continuous_grid_functions=model_funcs["continuous_grid_functions"],
        model_config=model_config,
    )

    states_and_assets_beginning_of_final_period, sim_dict = jax.lax.scan(
        f=simulate_body,
        init=states_initial_dtype,
        xs=sim_keys,
    )

    final_period_dict = simulate_final_period(
        states_and_assets_beginning_of_final_period,
        sim_keys=last_period_sim_keys,
        params=params,
        discrete_states_names=model_structure["discrete_states_names"],
        choice_range=model_structure["choice_range"],
        map_state_choice_to_index=jnp.asarray(
            model_structure["map_state_choice_to_index_with_proxy"]
        ),
        taste_shock_function=alt_model_funcs_sim["taste_shock_function"],
        compute_utility_final=model_funcs["compute_utility_final"],
        continuous_states_info=model_config["continuous_states_info"],
        model_structure_sol=model_structure,
        compute_assets_begin_of_period=alt_model_funcs_sim[
            "compute_assets_begin_of_period"
        ],
        budget_depends_on_choice=alt_model_funcs_sim["budget_depends_on_choice"],
        read_funcs=read_funcs,
    )

    # Standard simulation output

    result = {
        key: jnp.vstack([sim_dict[key], final_period_dict[key]])
        for key in sim_dict.keys()
        if key in final_period_dict.keys()
    }
    n_array_agents = jnp.ones(n_agents, dtype=float) * jnp.nan
    aux_results = {
        key: jnp.vstack([n_array_agents, sim_dict[key]])
        for key in sim_dict.keys()
        if key not in final_period_dict.keys()
    }
    result = {**result, **aux_results}
    return result


def simulate_single_period(
    states_beginning_of_period,
    sim_keys,
    params,
    endog_grid_solved,
    value_solved,
    policy_solved,
    model_structure_sol,
    model_funcs_sim,
    compute_utility,
    read_funcs,
    continuous_grid_functions,
    model_config,
):

    continuous_states_info = model_config["continuous_states_info"]
    has_additional_continuous_state = continuous_states_info[
        "has_additional_continuous_state"
    ]
    additional_continuous_state_names = continuous_states_info[
        "additional_continuous_state_names"
    ]

    # The carry holds bookkeeping entries alongside the states, so always select
    # the discrete states by name rather than taking the carry wholesale.
    discrete_states_beginning_of_period = {
        key: value
        for key, value in states_beginning_of_period.items()
        if key in model_structure_sol["discrete_states_names"]
    }
    choice_range = model_structure_sol["choice_range"]
    n_agents = states_beginning_of_period["assets_end_of_previous_period"].shape[0]

    # This period's own income shock, drawn here rather than inherited from last
    # period's transition, so the shock, the wealth it builds and the row it is
    # reported on all belong to the same period.
    income_shock = draw_income_shocks_for_period(
        n_agents=n_agents,
        params=params,
        read_funcs=read_funcs,
        income_shock_keys=sim_keys["income_shock_keys"],
    )

    if has_additional_continuous_state:
        # Already settled at the end of last period, unlike wealth: the continuous
        # state may not depend on the choice about to be made (see
        # _check_choice_not_declared in process_model_functions.py), so there is
        # nothing left to resolve here.
        continuous_state_beginning_of_period = {
            name: states_beginning_of_period[name]
            for name in additional_continuous_state_names
        }
    else:
        continuous_state_beginning_of_period = None

    # Law of motion at the *beginning* of the period, before any choice is made,
    # and evaluated for every choice: a budget equation may declare "choice" and
    # then hand the agent a different wealth per choice, which is exactly what the
    # solved solution is indexed by. Shape (n_agents, n_choices).
    assets_begin_of_period, budget_aux = assets_begin_of_period_for_each_choice(
        states_beginning_of_period=discrete_states_beginning_of_period,
        continuous_state_beginning_of_period=continuous_state_beginning_of_period,
        assets_end_of_previous_period=states_beginning_of_period[
            "assets_end_of_previous_period"
        ],
        income_shock=income_shock,
        choice_range=choice_range,
        params=params,
        compute_assets_begin_of_period=model_funcs_sim[
            "compute_assets_begin_of_period"
        ],
        budget_depends_on_choice=model_funcs_sim["budget_depends_on_choice"],
    )

    discount_factor = read_funcs["discount_factor"](params)
    # Interpolate policy and value function for all agents.
    policy, values_pre_taste_shock = interpolate_policy_and_value_for_all_agents(
        discrete_states_beginning_of_period=discrete_states_beginning_of_period,
        continuous_state_beginning_of_period=continuous_state_beginning_of_period,
        assets_begin_of_period=assets_begin_of_period,
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        map_state_choice_to_index=jnp.asarray(
            model_structure_sol["map_state_choice_to_index_with_proxy"]
        ),
        choice_range=model_structure_sol["choice_range"],
        params=params,
        discrete_states_names=model_structure_sol["discrete_states_names"],
        compute_utility=compute_utility,
        continuous_grid_functions=continuous_grid_functions,
        upper_envelope_method=model_config["upper_envelope"]["method"],
        has_additional_continuous_state=has_additional_continuous_state,
        discount_factor=discount_factor,
        skip_endog_grid_storage=model_config["upper_envelope"][
            "skip_endog_grid_storage"
        ],
    )

    # Draw taste shocks and calculate final value.
    taste_shocks = draw_taste_shocks(
        params=params,
        states_beginning_of_period=discrete_states_beginning_of_period,
        n_choices=len(choice_range),
        taste_shock_function=model_funcs_sim["taste_shock_function"],
        taste_shock_keys=sim_keys["taste_shock_keys"],
    )
    values_across_choices = values_pre_taste_shock + taste_shocks

    # Determine choice index of period by max value and select corresponding choice,
    # consumption and value.
    choice_index = jnp.nanargmax(values_across_choices, axis=1)
    choice = choice_range[choice_index]

    value_max = jnp.take_along_axis(
        values_across_choices, choice_index[:, None], axis=1
    )[:, 0]

    consumption = jnp.take_along_axis(policy, choice_index[:, None], axis=1)[:, 0]
    # The wealth the agent actually faces is the one belonging to the choice made.
    assets_realized = jnp.take_along_axis(
        assets_begin_of_period, choice_index[:, None], axis=1
    )[:, 0]
    budget_aux = {
        key: jnp.take_along_axis(val, choice_index[:, None], axis=1)[:, 0]
        for key, val in budget_aux.items()
    }
    if has_additional_continuous_state:
        states_of_period = {
            **discrete_states_beginning_of_period,
            **continuous_state_beginning_of_period,
        }
    else:
        states_of_period = discrete_states_beginning_of_period

    utility_period = vmap(vectorized_utility, in_axes=(0, 0, 0, None, None))(
        consumption,
        states_of_period,
        choice,
        params,
        compute_utility,
    )
    savings_current_period = assets_realized - consumption

    discrete_states_next_period, continuous_state_next_period = (
        transition_to_next_period(
            discrete_states_beginning_of_period=discrete_states_beginning_of_period,
            continuous_state_beginning_of_period=continuous_state_beginning_of_period,
            assets_end_of_period=savings_current_period,
            choice=choice,
            params=params,
            model_funcs_sim=model_funcs_sim,
            sim_keys=sim_keys,
        )
    )

    states_next_period = discrete_states_next_period

    # Carry what next period needs to run the budget equation at its beginning --
    # this period's end-of-period assets; it draws its own income shock -- alongside
    # next period's already-settled continuous state.
    if has_additional_continuous_state:
        for name in additional_continuous_state_names:
            states_next_period[name] = continuous_state_next_period[name]
    states_next_period["assets_end_of_previous_period"] = savings_current_period

    result = {
        "choice": choice,
        "consumption": consumption,
        "utility": utility_period,
        "taste_shocks": taste_shocks,
        "value_max": value_max,
        "value_choice": values_across_choices,
        "assets_begin_of_period": assets_realized,
        "savings": savings_current_period,
        "income_shock": income_shock,
        **budget_aux,
        # Only the model's own states -- the carry additionally holds a bookkeeping
        # entry (last period's end-of-period assets) that is not simulation output.
        **states_of_period,
    }

    return states_next_period, result


def simulate_final_period(
    states_begin_of_final_period,
    sim_keys,
    params,
    discrete_states_names,
    choice_range,
    map_state_choice_to_index,
    taste_shock_function,
    compute_utility_final,
    continuous_states_info,
    model_structure_sol,
    compute_assets_begin_of_period,
    budget_depends_on_choice,
    read_funcs,
):
    invalid_number = np.array(
        np.iinfo(map_state_choice_to_index.dtype).max,
        dtype=map_state_choice_to_index.dtype,
    )

    n_agents = len(states_begin_of_final_period["period"])
    discrete_states_begin_last_period = {
        key: value
        for key, value in states_begin_of_final_period.items()
        if key in model_structure_sol["discrete_states_names"]
    }
    # Drawn here, like every other period's (see simulate_single_period).
    income_shock_final_period = draw_income_shocks_for_period(
        n_agents=n_agents,
        params=params,
        read_funcs=read_funcs,
        income_shock_keys=sim_keys["income_shock_keys"],
    )

    # The final period applies the budget equation at its own beginning too, per
    # choice, exactly like every other period (see simulate_single_period). Its
    # continuous state arrives already settled from the previous period's
    # transition.
    if continuous_states_info["has_additional_continuous_state"]:
        continuous_state_begin_of_final_period = {
            name: states_begin_of_final_period[name]
            for name in continuous_states_info["additional_continuous_state_names"]
        }
    else:
        continuous_state_begin_of_final_period = None

    assets_begin_of_final_period, budget_aux_final = (
        assets_begin_of_period_for_each_choice(
            states_beginning_of_period=discrete_states_begin_last_period,
            continuous_state_beginning_of_period=continuous_state_begin_of_final_period,
            assets_end_of_previous_period=states_begin_of_final_period[
                "assets_end_of_previous_period"
            ],
            income_shock=income_shock_final_period,
            choice_range=choice_range,
            params=params,
            compute_assets_begin_of_period=compute_assets_begin_of_period,
            budget_depends_on_choice=budget_depends_on_choice,
        )
    )

    # Utility is evaluated per choice, at that choice's own wealth, so the state
    # dict handed to it is repeated across the choice axis. Every state in it is
    # choice-invariant -- only wealth differs by choice.
    n_choices = len(choice_range)
    states_begin_last_period = dict(discrete_states_begin_last_period)
    if continuous_state_begin_of_final_period is not None:
        states_begin_last_period.update(continuous_state_begin_of_final_period)
    states_per_choice = {
        key: jnp.repeat(val[:, None], n_choices, axis=1)
        for key, val in states_begin_last_period.items()
    }

    utilities_pre_taste_shock = vmap(
        vmap(
            compute_final_utility_for_each_choice,
            in_axes=(0, 0, 0, None, None),  # choices
        ),
        in_axes=(0, None, 0, None, None),  # agents
    )(
        states_per_choice,
        choice_range,
        assets_begin_of_final_period,
        params,
        compute_utility_final,
    )
    state_choice_indexes = get_state_choice_index_per_discrete_states(
        states=discrete_states_begin_last_period,
        map_state_choice_to_index=map_state_choice_to_index,
        discrete_states_names=discrete_states_names,
    )
    utilities_pre_taste_shock = jnp.where(
        state_choice_indexes == invalid_number, np.nan, utilities_pre_taste_shock
    )

    # Draw taste shocks and calculate final value.
    taste_shocks = draw_taste_shocks(
        params=params,
        states_beginning_of_period=discrete_states_begin_last_period,
        n_choices=len(choice_range),
        taste_shock_function=taste_shock_function,
        taste_shock_keys=sim_keys["taste_shock_keys"],
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

    # Everything is consumed in the final period, at the wealth belonging to the
    # choice actually made.
    assets_realized_final = jnp.take_along_axis(
        assets_begin_of_final_period, choice_index[:, None], axis=1
    )[:, 0]
    budget_aux_final = {
        key: jnp.take_along_axis(val, choice_index[:, None], axis=1)[:, 0]
        for key, val in budget_aux_final.items()
    }

    result = {
        "choice": choice,
        "consumption": assets_realized_final,
        "utility": utility_period,
        "value_max": value_period,
        "value_choice": values_across_choices[np.newaxis],
        "taste_shocks": taste_shocks[np.newaxis, :, :],
        "assets_begin_of_period": assets_realized_final,
        "savings": jnp.zeros_like(utility_period),
        "income_shock": income_shock_final_period,
        **budget_aux_final,
        **states_begin_last_period,
    }

    return result
