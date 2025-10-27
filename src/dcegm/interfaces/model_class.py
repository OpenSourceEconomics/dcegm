import pickle as pkl
from functools import partial
from typing import Callable, Dict

import jax
import pandas as pd

from dcegm.backward_induction import backward_induction
from dcegm.interfaces.index_functions import (
    get_child_state_index_per_states_and_choices,
    get_state_choice_index_per_discrete_states,
)
from dcegm.interfaces.interface import (
    get_n_state_choice_period,
    validate_stochastic_transition,
)
from dcegm.interfaces.sol_interface import model_solved
from dcegm.law_of_motion import calc_cont_grids_next_period
from dcegm.likelihood import create_individual_likelihood_function
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.alternative_sim_functions import (
    generate_alternative_sim_functions,
)
from dcegm.pre_processing.check_params import process_params
from dcegm.pre_processing.setup_model import (
    create_model_dict,
    create_model_dict_and_save,
    load_model_dict,
)
from dcegm.pre_processing.shared import try_jax_array
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
        model_save_path: str = None,
        model_load_path: str = None,
    ):
        """Setup the model and check if load or save is required."""

        if model_load_path is not None:
            model_dict = load_model_dict(
                model_config=model_config,
                model_specs=model_specs,
                utility_functions=utility_functions,
                utility_functions_final_period=utility_functions_final_period,
                budget_constraint=budget_constraint,
                state_space_functions=state_space_functions,
                stochastic_states_transitions=stochastic_states_transitions,
                shock_functions=shock_functions,
                path=model_load_path,
            )
        elif model_save_path is not None:
            model_dict = create_model_dict_and_save(
                model_config=model_config,
                model_specs=model_specs,
                utility_functions=utility_functions,
                utility_functions_final_period=utility_functions_final_period,
                budget_constraint=budget_constraint,
                state_space_functions=state_space_functions,
                stochastic_states_transitions=stochastic_states_transitions,
                shock_functions=shock_functions,
                path=model_save_path,
                debug_info=debug_info,
            )
        else:
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

        self.model_specs = jax.tree_util.tree_map(try_jax_array, model_specs)
        self.specs_without_jax = model_specs

        self.model_config = model_dict["model_config"]
        self.model_funcs = model_dict["model_funcs"]
        self.model_structure = model_dict["model_structure"]
        self.batch_info = model_dict["batch_info"]

        self.params_check_info = self.model_config["params_check_info"]

        income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
            model_config["n_quad_points"]
        )

        self.income_shock_draws_unscaled = income_shock_draws_unscaled
        self.income_shock_weights = income_shock_weights

        if alternative_sim_specifications is not None:
            self.alternative_sim_funcs = generate_alternative_sim_functions(
                model_specs=self.specs_without_jax,
                model_specs_jax=self.model_specs,
                **alternative_sim_specifications,
            )
        else:
            self.alternative_sim_funcs = None

    def set_alternative_sim_funcs(
        self, alternative_sim_specifications, alternative_specs=None
    ):
        if alternative_specs is None:
            self.alternative_sim_specs = self.model_specs
            alternative_specs_without_jax = self.specs_without_jax
        else:
            self.alternative_sim_specs = jax.tree_util.tree_map(
                try_jax_array, alternative_specs
            )
            alternative_specs_without_jax = alternative_specs

        alternative_sim_funcs = generate_alternative_sim_functions(
            model_specs=alternative_specs_without_jax,
            model_specs_jax=self.alternative_sim_specs,
            **alternative_sim_specifications,
        )
        self.alternative_sim_funcs = alternative_sim_funcs

    def backward_induction_inner_jit(self, params):
        return backward_induction(
            params=params,
            income_shock_draws_unscaled=self.income_shock_draws_unscaled,
            income_shock_weights=self.income_shock_weights,
            model_config=self.model_config,
            batch_info=self.batch_info,
            model_funcs=self.model_funcs,
            model_structure=self.model_structure,
        )

    def get_fast_solve_func(self):
        backward_jit = jax.jit(
            partial(
                backward_induction,
                income_shock_draws_unscaled=self.income_shock_draws_unscaled,
                income_shock_weights=self.income_shock_weights,
                model_config=self.model_config,
                batch_info=self.batch_info,
                model_funcs=self.model_funcs,
                model_structure=self.model_structure,
            )
        )

        return backward_jit

    def solve(self, params, load_sol_path=None, save_sol_path=None):
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

        params_processed = process_params(
            params, params_check_info=self.params_check_info
        )
        if load_sol_path is not None:
            sol_dict = pkl.load(open(load_sol_path, "rb"))
        else:
            value, policy, endog_grid = self.backward_induction_inner_jit(
                params_processed
            )
            sol_dict = {
                "value": value,
                "policy": policy,
                "endog_grid": endog_grid,
            }
            if save_sol_path is not None:
                pkl.dump(sol_dict, open(save_sol_path, "wb"))

        model_solved_class = model_solved(
            model=self,
            params=params,
            value=sol_dict["value"],
            policy=sol_dict["policy"],
            endog_grid=sol_dict["endog_grid"],
        )
        return model_solved_class

    def solve_and_simulate(
        self,
        params,
        states_initial,
        seed,
        load_sol_path=None,
        save_sol_path=None,
    ):
        """Solve the model and simulate it.

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
        params_processed = process_params(params, self.params_check_info)

        if load_sol_path is not None:
            sol_dict = pkl.load(open(load_sol_path, "rb"))
        else:
            value, policy, endog_grid = self.backward_induction_inner_jit(
                params_processed
            )

            sol_dict = {
                "value": value,
                "policy": policy,
                "endog_grid": endog_grid,
            }
            if save_sol_path is not None:
                pkl.dump(sol_dict, open(save_sol_path, "wb"))

        sim_dict = simulate_all_periods(
            states_initial=states_initial,
            n_periods=self.model_config["n_periods"],
            params=params,
            seed=seed,
            endog_grid_solved=sol_dict["endog_grid"],
            policy_solved=sol_dict["policy"],
            value_solved=sol_dict["value"],
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
        seed,
        slow_version=False,
    ):

        sim_func = lambda params, value, policy, endog_gid: simulate_all_periods(
            states_initial=states_initial,
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
            params_processed = process_params(params, self.params_check_info)
            # Solve the model
            value, policy, endog_grid = self.backward_induction_inner_jit(
                params_processed
            )

            sim_dict = sim_func(
                params=params_processed,
                value=value,
                policy=policy,
                endog_gid=endog_grid,
            )

            return sim_dict

        if slow_version:
            solve_simulate_func = solve_and_simulate_function_to_jit
        else:
            solve_simulate_func = jax.jit(solve_and_simulate_function_to_jit)

        def solve_and_simulate_function(params):
            sim_dict = solve_simulate_func(params)
            df = create_simulation_df(sim_dict)
            return df

        return solve_and_simulate_function

    def create_experimental_ll_func(
        self,
        params_all,
        observed_states,
        observed_choices,
        unobserved_state_specs=None,
        return_model_solution=False,
        use_probability_of_observed_states=True,
        slow_version=False,
    ):

        return create_individual_likelihood_function(
            model_structure=self.model_structure,
            model_config=self.model_config,
            model_funcs=self.model_funcs,
            model_specs=self.model_specs,
            backwards_induction_inner_jit=self.backward_induction_inner_jit,
            observed_states=observed_states,
            observed_choices=observed_choices,
            params_all=params_all,
            unobserved_state_specs=unobserved_state_specs,
            return_model_solution=return_model_solution,
            use_probability_of_observed_states=use_probability_of_observed_states,
            slow_version=slow_version,
        )

    def validate_exogenous(self, params):

        return validate_stochastic_transition(
            params=params,
            model_structure=self.model_structure,
            model_config=self.model_config,
            model_funcs=self.model_funcs,
        )

    def get_state_choices_idx(self, states):
        """Get the indices of the state choices for given states."""
        return get_state_choice_index_per_discrete_states(
            states=states,
            map_state_choice_to_index=self.model_structure["map_state_choice_to_index"],
            discrete_states_names=self.model_structure["discrete_states_names"],
        )

    def get_child_states(self, state, choice):
        if "map_state_choice_to_child_states" not in self.model_structure:
            raise ValueError(
                "For this function the model needs to be created with debug_info='all'"
            )

        child_idx = get_child_state_index_per_states_and_choices(
            states=state, choices=choice, model_structure=self.model_structure
        )
        state_space_dict = self.model_structure["state_space_dict"]
        discrete_states_names = self.model_structure["discrete_states_names"]
        child_states = {
            key: state_space_dict[key][child_idx] for key in discrete_states_names
        }
        return pd.DataFrame(child_states)

    def get_child_states_and_calc_trans_probs(self, state, choice, params):
        """Get the child states for a given state and choice and calculate the
        transition probabilities."""
        child_states_df = self.get_child_states(state, choice)

        trans_probs = self.model_funcs["compute_stochastic_transition_vec"](
            params=params, choice=choice, **state
        )
        child_states_df["trans_probs"] = trans_probs
        return child_states_df

    def get_full_child_states_by_asset_id_and_probs(
        self, state, choice, params, asset_id, second_continuous_id=None
    ):
        """Get the child states for a given state and choice and calculate the
        transition probabilities."""
        if "map_state_choice_to_child_states" not in self.model_structure:
            raise ValueError(
                "For this function the model needs to be created with debug_info='all'"
            )

        child_idx = get_child_state_index_per_states_and_choices(
            states=state, choices=choice, model_structure=self.model_structure
        )
        state_space_dict = self.model_structure["state_space_dict"]
        discrete_states_names = self.model_structure["discrete_states_names"]
        child_states = {
            key: state_space_dict[key][child_idx] for key in discrete_states_names
        }
        child_states_df = pd.DataFrame(child_states)

        child_continuous_states = self.compute_law_of_motions(params=params)

        if "second_continuous" in child_continuous_states.keys():
            if second_continuous_id is None:
                raise ValueError("second_continuous_id must be provided.")
            else:
                quad_wealth = child_continuous_states["assets_begin_of_period"][
                    child_idx, second_continuous_id, asset_id, :
                ]
                next_period_second_continuous = child_continuous_states[
                    "second_continuous"
                ][child_idx, second_continuous_id]

                second_continuous_name = self.model_config["continuous_states_info"][
                    "second_continuous_state_name"
                ]
                child_states_df[second_continuous_name] = next_period_second_continuous

        else:
            if second_continuous_id is not None:
                raise ValueError("second_continuous_id must not be provided.")
            else:
                quad_wealth = child_continuous_states["assets_begin_of_period"][
                    child_idx, asset_id, :
                ]

        for id_quad in range(quad_wealth.shape[1]):
            child_states_df[f"assets_begin_of_period_quad_point_{id_quad}"] = (
                quad_wealth[:, id_quad]
            )

        trans_probs = self.model_funcs["compute_stochastic_transition_vec"](
            params=params, choice=choice, **state
        )
        child_states_df["trans_probs"] = trans_probs
        return child_states_df

    def compute_law_of_motions(self, params):
        return calc_cont_grids_next_period(
            params=params,
            model_structure=self.model_structure,
            model_config=self.model_config,
            model_funcs=self.model_funcs,
            income_shock_draws_unscaled=self.income_shock_draws_unscaled,
        )

    def get_n_state_choices_per_period(self):
        return get_n_state_choice_period(self.model_structure)
