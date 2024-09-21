from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.final_periods import solve_final_period, solve_last_two_periods
from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp2d import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from dcegm.law_of_motion import (
    calculate_continuous_state,
    calculate_resources,
    calculate_resources_for_second_continuous_state,
)
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import create_solution_container, solve_dcegm
from dcegm.solve_single_period import solve_for_interpolated_values
from tests.utils.interp1d_auxiliary import (
    interpolate_policy_and_value_on_wealth_grid,
    linear_interpolation_with_extrapolation,
)
from toy_models.consumption_retirement_model.utility_functions import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)

N_PERIODS = 2
MAX_WEALTH = 50
WEALTH_GRID_POINTS = 100
EXPERIENCE_GRID_POINTS = 6


PARAMS = {
    "beta": 0.95,
    "delta": 0.35,
    "rho": 1.95,
    "savings_rate": 0.04,
    "constant": 0.75,
    "exp": 0.04,
    "exp_squared": -0.0002,
    "consumption_floor": 0.001,
}


# ====================================================================================
# Model functions
# ====================================================================================


def budget_constraint_continuous(
    period: int,
    lagged_choice: int,
    experience: float,
    savings_end_of_previous_period: float,
    income_shock_previous_period: float,
    options: Dict[str, Any],
    params: Dict[str, float],
) -> float:

    working = lagged_choice == 0

    experience_years = experience * period

    income_from_previous_period = _calc_stochastic_income(
        experience=experience_years,
        wage_shock=income_shock_previous_period,
        params=params,
    )

    wealth_beginning_of_period = (
        income_from_previous_period * working
        + (1 + params["interest_rate"]) * savings_end_of_previous_period
    )

    # Retirement safety net, only in retirement model, but we require to have it always
    # as a parameter
    return jnp.maximum(wealth_beginning_of_period, params["consumption_floor"])


def budget_constraint_discrete(
    lagged_choice: int,
    experience: int,
    savings_end_of_previous_period: float,
    income_shock_previous_period: float,
    options: Dict[str, Any],
    params: Dict[str, float],
) -> float:

    working = lagged_choice == 0

    income_from_previous_period = _calc_stochastic_income(
        experience=experience,
        wage_shock=income_shock_previous_period,
        params=params,
    )

    wealth_beginning_of_period = (
        income_from_previous_period * working
        + (1 + params["interest_rate"]) * savings_end_of_previous_period
    )

    # Retirement safety net, only in retirement model, but we require to have it always
    # as a parameter
    return jnp.maximum(wealth_beginning_of_period, params["consumption_floor"])


@jax.jit
def _calc_stochastic_income(
    experience: int,
    wage_shock: float,
    params: Dict[str, float],
) -> float:

    labor_income = (
        params["constant"]
        + params["exp"] * experience
        + params["exp_squared"] * experience**2
    )

    return jnp.exp(labor_income + wage_shock)


def get_next_period_experience(period, lagged_choice, experience, options, params):
    # ToDo: Rewrite in the sense of budget equation

    return (1 / period) * ((period - 1) * experience + (lagged_choice == 0))


def get_next_period_state(period, choice, experience):

    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    next_state["experience"] = experience + (choice == 0)

    return next_state


def get_next_period_discrete_state(period, choice):

    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    return next_state


def get_state_specific_feasible_choice_set(
    lagged_choice: int,
    options: Dict,
) -> np.ndarray:
    """Select state-specific feasible choice set such that retirement is absorbing."""

    n_choices = options["n_choices"]

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if lagged_choice == 1:
        feasible_choice_set = np.array([1])
    else:
        feasible_choice_set = np.arange(n_choices)

    return feasible_choice_set


def sparsity_condition(
    period,
    experience,
    options,
):

    max_init_experience = 0

    cond = True

    if (period + max_init_experience < experience) | (
        experience > options["n_periods"]
    ):
        cond = False

    return cond


# ====================================================================================
# Test
# ====================================================================================


def test_replication_discrete_versus_continuous_experience(load_example_model):
    options = {}
    model_name = "retirement_no_taste_shocks"
    params, _raw_options = load_example_model(f"{model_name}")

    options["model_params"] = _raw_options
    options["model_params"]["n_periods"] = N_PERIODS
    options["model_params"]["max_wealth"] = MAX_WEALTH
    options["model_params"]["n_grid_points"] = WEALTH_GRID_POINTS
    options["model_params"]["n_choices"] = _raw_options["n_discrete_choices"]

    options["state_space"] = {
        "n_periods": N_PERIODS,
        "choices": np.arange(2),
        "endogenous_states": {
            "experience": np.arange(N_PERIODS),
            "sparsity_condition": sparsity_condition,
        },
        "continuous_states": {
            "wealth": jnp.linspace(
                0,
                MAX_WEALTH,
                WEALTH_GRID_POINTS,
            )
        },
    }

    utility_functions = create_utility_function_dict()
    utility_functions_final_period = create_final_period_utility_function_dict()

    state_space_functions = {
        "get_next_period_state": get_next_period_state,
        "get_state_specific_feasible_choice_set": get_state_specific_feasible_choice_set,
    }

    model_disc = setup_model(
        options=options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_discrete,
    )

    (
        wealth_and_continuous_state_next_period_disc,
        taste_shock_scale,
        income_shock_weights,
        exog_grids_disc,
        wealth_beginning_at_regular_disc,
        model_funcs_disc,
        batch_info_disc,
        value_solved_disc,
        policy_solved_disc,
        endog_grid_solved_disc,
    ) = get_solve_last_two_periods_args(
        model_disc, params, has_second_continuous_state=False
    )

    # (
    #     value_disc,
    #     policy_disc,
    #     endog_grid_disc,
    # ) = solve_last_two_periods(
    #     wealth_and_continuous_state_next_period=wealth_and_continuous_state_next_period_disc,
    #     params=params,
    #     taste_shock_scale=taste_shock_scale,
    #     income_shock_weights=income_shock_weights,
    #     exog_grids=exog_grids_disc,
    #     wealth_beginning_at_regular=wealth_beginning_at_regular_disc,
    #     model_funcs=model_funcs_disc,
    #     batch_info=batch_info_disc,
    #     value_solved=value_solved_disc,
    #     policy_solved=policy_solved_disc,
    #     endog_grid_solved=endog_grid_solved_disc,
    #     has_second_continuous_state=False,
    # )

    (
        value_solved_disc,
        policy_solved_disc,
        endog_grid_solved_disc,
        value_last_regular_disc,
        marginal_utility_last_regular_disc,
        value_interp_final_period_disc,
        marginal_utility_final_last_period_disc,
    ) = solve_final_period(
        idx_state_choices_final_period=batch_info_disc[
            "idx_state_choices_final_period"
        ],
        idx_parent_states_final_period=batch_info_disc[
            "idxs_parent_states_final_period"
        ],
        state_choice_mat_final_period=batch_info_disc["state_choice_mat_final_period"],
        wealth_and_continuous_state_next_period=wealth_and_continuous_state_next_period_disc,
        wealth_beginning_at_regular_period=wealth_beginning_at_regular_disc,
        params=params,
        compute_utility=model_funcs_disc["compute_utility_final"],
        compute_marginal_utility=model_funcs_disc["compute_marginal_utility_final"],
        value_solved=value_solved_disc,
        policy_solved=policy_solved_disc,
        endog_grid_solved=endog_grid_solved_disc,
        has_second_continuous_state=False,  # since this is discrete, set to False
    )

    endog_grid_disc, policy_disc, value_disc = solve_for_interpolated_values(
        value_interpolated=value_interp_final_period_disc,
        marginal_utility_interpolated=marginal_utility_final_last_period_disc,
        state_choice_mat=batch_info_disc["state_choice_mat_second_last_period"],
        child_state_idxs=batch_info_disc["child_states_second_last_period"],
        states_to_choices_child_states=batch_info_disc["state_to_choices_final_period"],
        params=params,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
        exog_savings_grid=exog_grids_disc["wealth"],
        model_funcs=model_funcs_disc,
        has_second_continuous_state=False,  # For discrete case
    )

    idx_second_last_disc = batch_info_disc["idx_state_choices_second_last_period"]

    # To-Do: Second to last period not correct yet for second continuous case
    value_solved_disc = value_solved_disc.at[idx_second_last_disc, ...].set(value_disc)
    policy_solved_disc = policy_solved_disc.at[idx_second_last_disc, ...].set(
        policy_disc
    )
    endog_grid_solved_disc = endog_grid_solved_disc.at[idx_second_last_disc, ...].set(
        endog_grid_disc
    )

    # value_disc, policy_disc, endog_grid_disc = solve_dcegm(
    #     params,
    #     options,
    #     state_space_functions=state_space_functions,
    #     utility_functions=utility_functions,
    #     utility_functions_final_period=utility_functions_final_period,
    #     budget_constraint=budget_constraint_discrete,
    # )

    state_choice_space_disc = model_disc["model_structure"]["state_choice_space"]
    where_experience = model_disc["model_structure"]["state_space_names"].index(
        "experience"
    )

    # =================================================================================
    # Continuous experience
    # =================================================================================
    experience_grid = jnp.linspace(0, 1, EXPERIENCE_GRID_POINTS)

    options_cont = options.copy()
    options_cont["state_space"]["continuous_states"]["experience"] = experience_grid
    options_cont["state_space"]["endogenous_states"].pop("experience")
    options_cont["state_space"]["endogenous_states"].pop("sparsity_condition")

    state_space_functions_continuous = {
        "get_next_period_state": get_next_period_discrete_state,
        "update_continuous_state": get_next_period_experience,
        "get_state_specific_feasible_choice_set": get_state_specific_feasible_choice_set,
    }

    model_cont = setup_model(
        options=options_cont,
        state_space_functions=state_space_functions_continuous,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint_continuous,
    )
    # value_cont, policy_cont, endog_grid_cont = solve_dcegm(
    #     params,
    #     options_cont,
    #     state_space_functions=state_space_functions_continuous,
    #     utility_functions=utility_functions,
    #     utility_functions_final_period=utility_functions_final_period,
    #     budget_constraint=budget_constraint_continuous,
    # )

    (
        wealth_and_continuous_state_next_period_cont,
        taste_shock_scale,
        income_shock_weights,
        exog_grids_cont,
        wealth_beginning_at_regular_cont,
        model_funcs_cont,
        batch_info_cont,
        value_solved_cont,
        policy_solved_cont,
        endog_grid_solved_cont,
    ) = get_solve_last_two_periods_args(
        model_cont, params, has_second_continuous_state=True
    )

    # (
    #     value_cont,
    #     policy_cont,
    #     endog_grid_cont,
    # ) = solve_last_two_periods(
    #     wealth_and_continuous_state_next_period=wealth_and_continuous_state_next_period_cont,
    #     params=params,
    #     taste_shock_scale=taste_shock_scale,
    #     income_shock_weights=income_shock_weights,
    #     exog_grids=exog_grids_cont,
    #     wealth_beginning_at_regular=wealth_beginning_at_regular_cont,
    #     model_funcs=model_funcs_cont,
    #     batch_info=batch_info_cont,
    #     value_solved=value_solved_cont,
    #     policy_solved=policy_solved_cont,
    #     endog_grid_solved=endog_grid_solved_cont,
    #     has_second_continuous_state=True,
    # )

    (
        value_solved_cont,
        policy_solved_cont,
        endog_grid_solved_cont,
        value_last_regular_cont,
        marginal_utility_last_regular_cont,
        value_interp_final_period_cont,
        marginal_utility_final_last_period_cont,
    ) = solve_final_period(
        idx_state_choices_final_period=batch_info_cont[
            "idx_state_choices_final_period"
        ],
        idx_parent_states_final_period=batch_info_cont[
            "idxs_parent_states_final_period"
        ],
        state_choice_mat_final_period=batch_info_cont["state_choice_mat_final_period"],
        wealth_and_continuous_state_next_period=wealth_and_continuous_state_next_period_cont,
        wealth_beginning_at_regular_period=wealth_beginning_at_regular_cont,
        params=params,
        compute_utility=model_funcs_cont["compute_utility_final"],
        compute_marginal_utility=model_funcs_cont["compute_marginal_utility_final"],
        value_solved=value_solved_cont,
        policy_solved=policy_solved_cont,
        endog_grid_solved=endog_grid_solved_cont,
        has_second_continuous_state=True,  # since this is continuous, set to True
    )

    idx_state_choices_final_period = batch_info_cont["idx_state_choices_final_period"]
    idx_parent_states_final_period = batch_info_cont["idxs_parent_states_final_period"]

    _continuous_state, resources = wealth_and_continuous_state_next_period_cont
    # continuous_state = _continuous_state[idx_parent_states_final_period]
    resources = resources[idx_parent_states_final_period]
    n_wealth = resources.shape[2]

    value_solved_last_regular = value_solved_cont[
        idx_state_choices_final_period, :, 1 : n_wealth + 1
    ]
    policy_solved_last_regular = policy_solved_cont[
        idx_state_choices_final_period, :, 1 : n_wealth + 1
    ]
    endog_grid_solved_last_regular = endog_grid_solved_cont[
        idx_state_choices_final_period, :, 1 : n_wealth + 1
    ]
    marg_util_hand_cont = policy_solved_last_regular ** (-params["rho"])

    endog_grid_cont, policy_cont, value_cont = solve_for_interpolated_values(
        value_interpolated=value_interp_final_period_cont,
        marginal_utility_interpolated=marginal_utility_final_last_period_cont,
        state_choice_mat=batch_info_cont["state_choice_mat_second_last_period"],
        child_state_idxs=batch_info_cont["child_states_second_last_period"],
        states_to_choices_child_states=batch_info_cont["state_to_choices_final_period"],
        params=params,
        taste_shock_scale=taste_shock_scale,
        income_shock_weights=income_shock_weights,
        exog_savings_grid=exog_grids_cont["wealth"],
        model_funcs=model_funcs_cont,
        has_second_continuous_state=True,  # For continuous case
    )

    idx_second_last_cont = batch_info_cont["idx_state_choices_second_last_period"]

    # To-Do: Second to last period not correct yet for second continuous case
    value_solved_cont = value_solved_cont.at[idx_second_last_cont, ...].set(value_cont)
    policy_solved_cont = policy_solved_cont.at[idx_second_last_cont, ...].set(
        policy_cont
    )
    endog_grid_solved_cont = endog_grid_solved_cont.at[idx_second_last_cont, ...].set(
        endog_grid_cont
    )

    state_choice_space_cont = model_cont["model_structure"]["state_choice_space"]

    # =================================================================================
    # Interpolate
    # =================================================================================

    period = 0
    experience = 0
    # exp_share_to_test = experience / period
    exp_share_to_test = 0

    state_choice_disc = state_choice_space_disc[
        (state_choice_space_disc[:, 0] == period)
        & (state_choice_space_disc[:, where_experience] == experience)
    ][-1]
    state_choice_cont = state_choice_space_cont[
        state_choice_space_cont[:, 0] == period
    ][-1]

    idx_disc = jnp.where(jnp.all(state_choice_space_disc == state_choice_disc, axis=1))[
        0
    ]
    idx_cont = jnp.where(jnp.all(state_choice_space_cont == state_choice_cont, axis=1))[
        0
    ]

    state_space_names_disc = model_disc["model_structure"]["state_space_names"]
    state_space_names_disc.append("choice")
    state_choice_vec_disc = dict(zip(state_space_names_disc, state_choice_disc))

    state_space_names_cont = model_cont["model_structure"]["state_space_names"]
    state_space_names_cont.append("choice")
    state_choice_vec_cont = dict(zip(state_space_names_cont, state_choice_cont))

    for wealth_to_test in np.arange(5, 100, 5, dtype=float):

        policy_cont_interp, value_cont_interp = (
            interp2d_policy_and_value_on_wealth_and_regular_grid(
                regular_grid=experience_grid,
                wealth_grid=jnp.squeeze(endog_grid_cont[idx_cont], axis=0),
                policy_grid=jnp.squeeze(policy_cont[idx_cont], axis=0),
                value_grid=jnp.squeeze(value_cont[idx_cont], axis=0),
                regular_point_to_interp=exp_share_to_test,
                wealth_point_to_interp=jnp.array(wealth_to_test),
                compute_utility=model_cont["model_funcs"]["compute_utility"],
                state_choice_vec=state_choice_vec_cont,
                params=params,
            )
        )

        policy_disc_interp, value_disc_interp = interp1d_policy_and_value_on_wealth(
            wealth=jnp.array(wealth_to_test),
            endog_grid=jnp.squeeze(endog_grid_disc[idx_disc], axis=0),
            policy=jnp.squeeze(policy_disc[idx_disc], axis=0),
            value=jnp.squeeze(value_disc[idx_disc], axis=0),
            compute_utility=model_disc["model_funcs"]["compute_utility"],
            state_choice_vec=state_choice_vec_disc,
            params=params,
        )

        aaae(value_cont_interp, value_disc_interp)
        aaae(policy_cont_interp, policy_disc_interp)


def get_solve_last_two_periods_args(model, params, has_second_continuous_state):
    options = model["options"]
    batch_info = model["batch_info"]

    exog_grids = options["exog_grids"]

    # Prepare income shock draws and scaling
    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        options["model_params"]["quadrature_points_stochastic"]
    )
    taste_shock_scale = params["lambda"]

    # Get state space dictionary and model functions
    state_space_dict = model["model_structure"]["state_space_dict"]
    model_funcs = model["model_funcs"]

    # Distinction based on has_second_continuous_state
    if has_second_continuous_state:
        # Calculate continuous state for the next period
        continuous_state_next_period = calculate_continuous_state(
            discrete_states_beginning_of_period=state_space_dict,
            continuous_grid=exog_grids["second_continuous"],
            params=params,
            compute_continuous_state=model_funcs[
                "compute_beginning_of_period_continuous_state"
            ],
        )

        # Extra dimension for continuous state
        wealth_beginning_of_next_period = (
            calculate_resources_for_second_continuous_state(
                discrete_states_beginning_of_next_period=state_space_dict,
                continuous_state_beginning_of_next_period=continuous_state_next_period,
                savings_grid=exog_grids["wealth"],
                income_shocks=income_shock_draws_unscaled * params["sigma"],
                params=params,
                compute_beginning_of_period_resources=model_funcs[
                    "compute_beginning_of_period_resources"
                ],
            )
        )

        # Extra dimension for continuous state (regular wealth calculation)
        wealth_beginning_at_regular = calculate_resources_for_second_continuous_state(
            discrete_states_beginning_of_next_period=state_space_dict,
            continuous_state_beginning_of_next_period=jnp.tile(
                exog_grids["second_continuous"],
                (continuous_state_next_period.shape[0], 1),
            ),
            savings_grid=exog_grids["wealth"],
            income_shocks=income_shock_draws_unscaled * params["sigma"],
            params=params,
            compute_beginning_of_period_resources=model_funcs[
                "compute_beginning_of_period_resources"
            ],
        )

        # Combined wealth and continuous state for the next period
        wealth_and_continuous_state_next_period = (
            continuous_state_next_period,
            wealth_beginning_of_next_period,
        )

    else:
        # Single state calculation (no second continuous state)
        wealth_and_continuous_state_next_period = calculate_resources(
            discrete_states_beginning_of_period=state_space_dict,
            savings_grid=exog_grids["wealth"],
            income_shocks_current_period=income_shock_draws_unscaled * params["sigma"],
            params=params,
            compute_beginning_of_period_resources=model_funcs[
                "compute_beginning_of_period_resources"
            ],
        )
        wealth_beginning_at_regular = None

    # Create solution containers for value, policy, and endogenous grids
    value_solved, policy_solved, endog_grid_solved = create_solution_container(
        n_state_choices=model["model_structure"]["state_choice_space"].shape[0],
        options=options,
        has_second_continuous_state=has_second_continuous_state,
    )

    # Return the relevant arguments as a tuple
    return (
        wealth_and_continuous_state_next_period,
        taste_shock_scale,
        income_shock_weights,
        exog_grids,
        wealth_beginning_at_regular,
        model_funcs,
        batch_info,
        value_solved,
        policy_solved,
        endog_grid_solved,
    )
