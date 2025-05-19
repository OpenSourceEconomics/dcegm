from functools import partial
from typing import Callable, Dict

import jax

from dcegm.backward_induction import backward_induction
from dcegm.interfaces.sol_interface import model_solved
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.alternative_sim_functions import (
    generate_alternative_sim_functions,
)
from dcegm.pre_processing.check_params import process_params
from dcegm.pre_processing.setup_model import create_model_dict
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods


class setup_model:
    def __init__(
        self,
        model_config: Dict,
        model_specs: Dict,
        utility_functions: Dict[str, Callable],
        utility_functions_final_period: Dict[str, Callable],
        budget_constraint: Callable,
        state_space_functions: Dict[str, Callable] = None,
        stochastic_states_transitions: Dict[str, Callable] = None,
        shock_functions: Dict[str, Callable] = None,
        alternative_sim_specifications: Dict[str, Callable] = None,
        debug_info: str = None,
    ):
        model_dict = create_model_dict(
            model_config=model_config,
            model_specs=model_specs,
            utility_functions=utility_functions,
            utility_functions_final_period=utility_functions_final_period,
            budget_constraint=budget_constraint,
            state_space_functions=state_space_functions,
            stochastic_states_transitions=stochastic_states_transitions,
            shock_functions=shock_functions,
            debug_info=debug_info,
        )

        self.model_config = model_dict["model_config"]
        self.model_funcs = model_dict["model_funcs"]
        self.model_structure = model_dict["model_structure"]
        self.batch_info = model_dict["batch_info"]

        income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
            model_config["n_quad_points"]
        )

        self.income_shock_draws_unscaled = income_shock_draws_unscaled
        self.income_shock_weights = income_shock_weights

        has_second_continuous_state = self.model_config["continuous_states_info"][
            "second_continuous_exists"
        ]

        backward_jit = jax.jit(
            partial(
                backward_induction,
                model_config=self.model_config,
                has_second_continuous_state=has_second_continuous_state,
                state_space_dict=self.model_structure["state_space_dict"],
                n_state_choices=self.model_structure["state_choice_space"].shape[0],
                batch_info=self.batch_info,
                income_shock_draws_unscaled=self.income_shock_draws_unscaled,
                income_shock_weights=self.income_shock_weights,
                model_funcs=self.model_funcs,
            )
        )

        self.backward_induction_jit = backward_jit

        if alternative_sim_specifications is not None:
            self.alternative_sim_funcs = generate_alternative_sim_functions(
                **alternative_sim_specifications
            )
        else:
            self.alternative_sim_funcs = None

    def solve(self, params):
        """Solve a discrete-continuous life-cycle model using the DC-EGM algorithm.

        Args:
            params (pd.DataFrame): Params DataFrame.
            options (dict): Options dictionary.
            utility_functions (Dict[str, callable]): Dictionary of three user-supplied
                functions for computation of:
                (i) utility
                (ii) inverse marginal utility
                (iii) next period marginal utility
            budget_constraint (callable): Callable budget constraint.
            state_space_functions (Dict[str, callable]): Dictionary of two user-supplied
                functions to:
                (i) get the state specific feasible choice set
                (ii) update the endogenous part of the state by the choice
            final_period_solution (callable): User-supplied function for solving the agent's
                last period.
            transition_function (callable): User-supplied function returning for each
                state a transition matrix vector.

        """
        params_processed = process_params(params)
        # Solve the model
        value, policy, endog_grid = self.backward_induction_jit(params_processed)

        model_solved_class = model_solved(
            value=value,
            policy=policy,
            endog_grid=endog_grid,
            model_config=self.model_config,
            model_structure=self.model_structure,
            model_funcs=self.model_funcs,
            params=params_processed,
        )
        return model_solved_class

    def solve_and_simulate(
        self,
        params,
        states_initial,
        seed,
    ):
        """
        Solve the model and simulate it.

        Args:
            params: The parameters for the model.
            states_initial: The initial states for the simulation.
            wealth_initial: The initial wealth for the simulation.
            n_periods: The number of periods to simulate.
            seed: The random seed for the simulation.
            alt_model_funcs_sim: Alternative model functions for simulation.

        Returns:
            A dictionary containing the solution and simulation results.
        """
        params_processed = process_params(params)

        value, policy, endog_grid = self.backward_induction_jit(params_processed)

        sim_dict = simulate_all_periods(
            states_initial=states_initial,
            n_periods=self.model_config["n_periods"],
            params=params,
            seed=seed,
            endog_grid_solved=endog_grid,
            policy_solved=policy,
            value_solved=value,
            model_config=self.model_config,
            model_structure=self.model_structure,
            model_funcs=self.model_funcs,
            alt_model_funcs_sim=self.alternative_sim_funcs,
        )

        sim_df = create_simulation_df(sim_dict)
        return sim_df

    def get_solve_and_simulate_func(
        self,
        states_initial,
        wealth_initial,
        seed,
    ):

        sim_func = lambda params, value, policy, endog_gid: simulate_all_periods(
            states_initial=states_initial,
            wealth_initial=wealth_initial,
            n_periods=self.model_config["n_periods"],
            params=params,
            seed=seed,
            endog_grid_solved=endog_gid,
            policy_solved=policy,
            value_solved=value,
            model_config=self.model_config,
            model_structure=self.model_structure,
            model_funcs=self.model_funcs,
            alt_model_funcs_sim=self.alternative_sim_funcs,
        )

        def solve_and_simulate_function_to_jit(params):
            params_processed = process_params(params)
            # Solve the model
            value, policy, endog_grid = self.backward_induction_jit(params_processed)

            sim_dict = sim_func(
                params=params_processed,
                value=value,
                policy=policy,
                endog_gid=endog_grid,
            )

            return sim_dict

        jit_solve_simulate = jax.jit(solve_and_simulate_function_to_jit)

        def solve_and_simulate_function(params):
            sim_dict = jit_solve_simulate(params)
            df = create_simulation_df(sim_dict)
            return df

        return solve_and_simulate_function
