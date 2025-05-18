import jax.numpy as jnp
import numpy as np
import pytest

import dcegm.toy_models as toy_models
from dcegm.pre_processing.alternative_sim_functions import (
    generate_alternative_sim_functions,
)
from dcegm.pre_processing.setup_model import setup_model
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.solve import get_solve_func_for_model


def utility_crra(
    consumption,
    choice,
    married,
    params,
):

    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])

    utility = (
        utility_consumption
        + married * params["married_util"]
        - (1 - choice) * params["delta"]
    )

    return utility


def marriage_transition(married, model_specs):
    trans_mat = model_specs["marriage_trans_mat"]
    return trans_mat[married, :]


@pytest.fixture
def model_configs():
    model_config_sol = {
        "n_periods": 5,
        "choices": np.arange(2),
        "endogenous_states": {
            "married": np.arange(2, dtype=int),
        },
        "continuous_states": {
            "wealth": np.arange(0, 100, 5, dtype=float),
        },
    }

    model_config_sim = {
        "n_periods": 5,
        "choices": np.arange(2),
        "exogenous_processes": {"married": np.arange(2, dtype=int)},
        "continuous_states": {
            "wealth": np.arange(0, 100, 5, dtype=float),
        },
    }

    model_configs = {
        "solution": model_config_sol,
        "simulation": model_config_sim,
    }

    return model_configs


def test_sim_and_sol_model(model_configs):
    params, model_specs, _ = toy_models.load_example_params_model_specs_and_config(
        "dcegm_paper_retirement_with_shocks"
    )
    params["married_util"] = 0.5

    model_funcs = toy_models.load_example_model_functions("dcegm_paper")
    utility_functions = model_funcs["utility_functions"]
    utility_functions["utility"] = utility_crra

    model_sol = setup_model(
        model_config=model_configs["solution"],
        model_specs=model_specs,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=utility_functions,
        utility_functions_final_period=model_funcs["utility_functions_final_period"],
        budget_constraint=model_funcs["budget_constraint"],
    )
    solve_func = get_solve_func_for_model(model_sol)

    value, policy, endog_grid = solve_func(params)

    marriage_trans_mat = jnp.array([[0.3, 0.7], [0.1, 0.9]])
    model_specs["marriage_trans_mat"] = marriage_trans_mat

    exogenous_states_transitions = {"marriage_transition": marriage_transition}

    alt_model_funcs_sim = generate_alternative_sim_functions(
        model_config=model_configs["simulation"],
        model_specs=model_specs,
        state_space_functions=model_funcs["state_space_functions"],
        budget_constraint=model_funcs["budget_constraint"],
        exogenous_states_transitions=exogenous_states_transitions,
    )

    n_agents = 100_000
    initial_marriage_dist = np.array([0.5, 0.5])
    single_states = np.zeros(int(n_agents / 2), dtype=int)
    married_states = np.ones(int(n_agents / 2), dtype=int)
    initial_marriage_states = np.concatenate([single_states, married_states])

    states_initial = {
        "period": np.zeros(n_agents, dtype=int),
        "lagged_choice": np.zeros(n_agents, dtype=int),
        "married": initial_marriage_states,
    }
    n_periods = model_configs["simulation"]["n_periods"]

    sim_dict = simulate_all_periods(
        states_initial=states_initial,
        wealth_initial=np.ones(n_agents, dtype=float) * 10,
        n_periods=n_periods,
        params=params,
        seed=123,
        endog_grid_solved=endog_grid,
        policy_solved=policy,
        value_solved=value,
        model=model_sol,
        alt_model_funcs_sim=alt_model_funcs_sim,
    )
    df = create_simulation_df(sim_dict)

    ###########################################
    # Compare marriage shares as they must be governed
    # by the transition matrix in the simulation
    ###########################################

    marriage_shares_sim = df.groupby("period")["married"].value_counts(normalize=True)

    atol_marriage = 1e-2
    # We compare married shares, as single shares are just 1 - married shares
    np.testing.assert_allclose(
        marriage_shares_sim.loc[(0, 1)],
        initial_marriage_dist[1],
        atol=atol_marriage,
    )

    for period in range(1, n_periods):
        last_period_marriage_shares = marriage_shares_sim.loc[
            (period - 1, [0, 1])
        ].values
        predicted_shares = last_period_marriage_shares @ marriage_trans_mat
        np.testing.assert_allclose(
            marriage_shares_sim.loc[(period, 1)],
            predicted_shares[1],
            atol=atol_marriage,
        )

    ###########################################
    # Compare values of the never married and all time married
    # with the values from the solution
    ###########################################

    _cond = [df["choice"] == 0, df["choice"] == 1]
    _val = [df["taste_shocks_0"], df["taste_shocks_1"]]
    df["taste_shock_realized_of_expected"] = np.select(_cond, _val)

    df["discount_factors"] = params["beta"] ** df.index.get_level_values("period")
    df["disc_expected_utility"] = df["discount_factors"] * (
        df["taste_shock_realized_of_expected"] + df["utility"]
    )
    # Finally discounted utility sum by agent
    disc_util_sums = df.groupby("agent")["disc_expected_utility"].sum()
    # Create sum of married variable to select individuals who where always married
    df["married_sum"] = df.groupby("agent")["married"].transform("sum")

    for married_sum in [0, 5]:
        df_always = df[df["married_sum"] == married_sum]

        # Now compare value of always married individuals
        for choice in [0, 1]:
            # Select all individuals who made the choice in the first period
            # Their expected and realized values of the choice should be the same
            df_always_period_0 = df_always.loc[(0, slice(None))]
            df_always_period_0_choice = df_always_period_0[
                df_always_period_0["choice"] == choice
            ]
            if df_always_period_0_choice.shape[0] > 0:
                relevant_agents = df_always_period_0_choice.index.get_level_values(
                    "agent"
                )
                realized_value = disc_util_sums[relevant_agents].mean()
                # In their expected value, we also have the realization of the first taste shock,
                # but as they choose 0 this is also the realized taste shock used in the realized value
                expected_value = df_always.loc[
                    (0, relevant_agents), f"value_choice_{choice}"
                ].mean()
                np.testing.assert_allclose(
                    realized_value,
                    expected_value,
                    atol=1e-1,
                )
