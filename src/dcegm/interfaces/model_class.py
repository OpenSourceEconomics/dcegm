import pickle as pkl
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import pandas as pd
from jax import numpy as jnp

from dcegm.backward_induction import backward_induction
from dcegm.interfaces.index_functions import (
    get_child_state_index_per_states_and_choices,
    get_state_choice_index_per_discrete_states,
)
from dcegm.interfaces.inspect_solution import partially_solve
from dcegm.interfaces.interface import (
    get_n_state_choice_period,
    validate_stochastic_transition,
)
from dcegm.interfaces.jit_large_arrays import (
    merg_non_jit_batch_info_and_jit_batch_info,
    merge_non_jit_and_jit_model_structure,
    split_structure_and_batch_info,
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
    """User-facing entry point: builds a discrete-continuous life-cycle model.

    Constructing this class runs the full model-build pipeline once (state space, state-
    choice space, batching for backward induction) and caches the result on the instance
    as ``model_config``, ``model_funcs``, ``model_structure`` and ``batch_info``. Every
    other method -- ``solve``, ``solve_and_simulate``, the ``get_*_func`` jit-compiling
    factories, the likelihood-function factory, and the debug/inspection helpers --
    reuses that cached build rather than repeating it.

    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        model_specs: Dict[str, Any],
        utility_functions: Dict[str, Callable],
        utility_functions_final_period: Dict[str, Callable],
        budget_constraint: Callable,
        state_space_functions: Optional[Dict[str, Callable]] = None,
        stochastic_states_transitions: Optional[Dict[str, Callable]] = None,
        shock_functions: Optional[Dict[str, Callable]] = None,
        continuous_grid_functions: Optional[Dict[str, Callable]] = None,
        alternative_sim_specifications: Optional[Dict[str, Callable]] = None,
        debug_info: Optional[str] = None,
        model_save_path: Optional[str] = None,
        model_load_path: Optional[str] = None,
        use_stochastic_sparsity: bool = False,
    ) -> None:
        """Build (or load) the model and cache its structure on the instance.

        Exactly one of ``model_load_path``/``model_save_path`` may be given: if
        ``model_load_path`` is set, the model is read back from disk instead of
        rebuilt; if ``model_save_path`` is set, it is built as usual and then
        also written to disk for a later ``model_load_path`` call. If neither
        is given, the model is built fresh and kept only in memory.

        Args:
            model_config: Model configuration -- periods, choices, the
                continuous grids, the upper-envelope method, and related
                solver settings.
            model_specs: User-supplied numerical model specifications (e.g.
                economic parameters that are fixed across estimation, unlike
                ``params``). Converted to a jax pytree via ``try_jax_array``.
            utility_functions: User-supplied utility, marginal utility, and
                inverse marginal utility functions for the regular periods.
            utility_functions_final_period: The same, for the final period
                (typically a simplified "consume everything" problem).
            budget_constraint: User-supplied callable computing beginning-of-
                period assets from the previous period's choice and shocks.
            state_space_functions: Optional user-supplied functions for (i)
                the state-specific feasible choice set and (ii) the
                endogenous state update given a choice.
            stochastic_states_transitions: Optional user-supplied transition
                functions for stochastic (exogenous) discrete states.
            shock_functions: Optional user-supplied taste-shock functions.
            continuous_grid_functions: Optional user-supplied functions
                returning (possibly state-specific) continuous grids, keyed by
                grid name.
            alternative_sim_specifications: Optional alternative simulation
                functions, used to build ``self.alternative_sim_funcs`` via
                ``generate_alternative_sim_functions``. If omitted,
                ``self.alternative_sim_funcs`` is ``None`` and simulation
                falls back to ``model_funcs``.
            debug_info: Optional flag controlling how much extra (non-jit-
                friendly) debug information the build keeps around, e.g.
                ``"all"`` to keep ``map_state_choice_to_child_states`` for
                ``get_child_states``.
            model_save_path: Optional path to save the built model dict to.
            model_load_path: Optional path to load a previously saved model
                dict from, skipping the build.
            use_stochastic_sparsity: EXPERIMENTAL: use stochastic transition
                sparsity.

        """
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
                continuous_grid_functions=continuous_grid_functions,
                path=model_load_path,
                use_stochastic_sparsity=use_stochastic_sparsity,
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
                continuous_grid_functions=continuous_grid_functions,
                path=model_save_path,
                debug_info=debug_info,
                use_stochastic_sparsity=use_stochastic_sparsity,
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
                continuous_grid_functions=continuous_grid_functions,
                debug_info=debug_info,
                use_stochastic_sparsity=use_stochastic_sparsity,
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

    def solve(
        self,
        params: Dict[str, jnp.ndarray],
        load_sol_path: Optional[str] = None,
        save_sol_path: Optional[str] = None,
    ) -> model_solved:
        """Solve the model via backward induction for one set of parameters.

        Not jit-compiled -- each call retraces ``backward_induction`` from
        scratch. For repeated solves at different parameters (e.g. inside an
        optimizer or MCMC sampler), use ``get_solve_func`` instead, which
        compiles once and reuses the compiled function.

        Args:
            params: Model parameters.
            load_sol_path: Optional path to a pickled solution dict
                (``{"value", "policy", "endog_grid"}``); if given, the model is
                not solved and this solution is loaded instead.
            save_sol_path: Optional path to pickle the freshly computed
                solution dict to. Ignored when ``load_sol_path`` is given.

        Returns:
            The solved model, wrapping ``value``, ``policy`` and
            ``endog_grid`` together with this model's structure for
            downstream inspection/simulation.

        """
        params_processed = process_params(
            params, params_check_info=self.params_check_info
        )
        if load_sol_path is not None:
            sol_dict = pkl.load(open(load_sol_path, "rb"))
        else:
            value, policy, endog_grid = backward_induction(
                params=params_processed,
                income_shock_draws_unscaled=self.income_shock_draws_unscaled,
                income_shock_weights=self.income_shock_weights,
                model_config=self.model_config,
                model_funcs=self.model_funcs,
                model_structure=self.model_structure,
                batch_info=self.batch_info,
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
        params: Dict[str, jnp.ndarray],
        states_initial: Dict[str, jnp.ndarray],
        seed: int,
        load_sol_path: Optional[str] = None,
        save_sol_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Solve the model, then simulate a panel of agents forward through it.

        Args:
            params: Model parameters, in any form ``process_params`` accepts.
            states_initial: Initial discrete (and, if applicable, continuous)
                states for the simulated agents, one array per state name,
                each of shape ``(n_agents,)``.
            seed: Random seed for the simulation's taste shocks and income
                draws.
            load_sol_path: Optional path to a pickled solution dict to load
                instead of solving.
            save_sol_path: Optional path to pickle the freshly computed
                solution dict to. Ignored when ``load_sol_path`` is given.

        Returns:
            A long-format panel with a ``(period, agent)`` MultiIndex,
            containing simulated states, choices, consumption, and value.

        """
        params_processed = process_params(params, self.params_check_info)

        if load_sol_path is not None:
            sol_dict = pkl.load(open(load_sol_path, "rb"))
        else:
            value, policy, endog_grid = backward_induction(
                params=params_processed,
                income_shock_draws_unscaled=self.income_shock_draws_unscaled,
                income_shock_weights=self.income_shock_weights,
                model_config=self.model_config,
                model_funcs=self.model_funcs,
                model_structure=self.model_structure,
                batch_info=self.batch_info,
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

    def get_solve_func(self) -> Callable:
        """Build a solve function that is jit-compiled once, on its first call.

        The (large, array-valued) parts of ``model_structure``/``batch_info``
        are split off via ``split_structure_and_batch_info`` and threaded
        through ``jax.jit`` as traced arguments; the remaining, non-jit-
        friendly parts are closed over as static Python objects. Subsequent
        calls at different ``params`` reuse the same compiled XLA executable.

        Returns:
            A function ``solve_function(params) -> model_solved`` with the
            same solving semantics as ``solve`` (without the load/save
            options), but compiled.

        """
        (
            model_structure_for_jit,
            batch_info_for_jit,
            model_structure_non_jit,
            batch_info_non_jit,
        ) = split_structure_and_batch_info(self.model_structure, self.batch_info)

        def solve_function_to_jit(params, model_structure_jit, batch_info_jit):
            params_processed = process_params(params, self.params_check_info)

            # Merge back parts together. The non_jit objects are fixed in the closure.
            model_structure = merge_non_jit_and_jit_model_structure(
                model_structure_jit, model_structure_non_jit
            )
            batch_info = merg_non_jit_batch_info_and_jit_batch_info(
                batch_info_jit, batch_info_non_jit
            )

            # Solve the model.
            value, policy, endog_grid = backward_induction(
                params=params_processed,
                model_structure=model_structure,
                batch_info=batch_info,
                income_shock_draws_unscaled=self.income_shock_draws_unscaled,
                income_shock_weights=self.income_shock_weights,
                model_config=self.model_config,
                model_funcs=self.model_funcs,
            )

            return value, policy, endog_grid

        solve_func = jax.jit(solve_function_to_jit)

        # Generate the function. The user only needs to provide params, but we call with the objects for jit.
        def solve_function(params: Dict[str, jnp.ndarray]) -> model_solved:
            """Solve the model for given params."""
            value, policy, endog_grid = solve_func(
                params, model_structure_for_jit, batch_info_for_jit
            )
            model_solved_class = model_solved(
                model=self,
                params=params,
                value=value,
                policy=policy,
                endog_grid=endog_grid,
            )
            return model_solved_class

        return solve_function

    def get_solve_and_simulate_func(
        self,
        states_initial: Dict[str, jnp.ndarray],
        seed: int,
        slow_version: bool = False,
    ) -> Callable[[Dict[str, jnp.ndarray]], pd.DataFrame]:
        """Build a solve-and-simulate function, jit-compiled on its first call.

        Same jit strategy as ``get_solve_func`` (splitting ``model_structure``/
        ``batch_info`` into traced vs. static parts), but the returned
        function also simulates a panel from the solution before returning,
        in a single compiled call.

        Args:
            states_initial: Initial states for the simulated agents; fixed for
                every call of the returned function (only ``params`` varies).
            seed: Random seed for the simulation; likewise fixed.
            slow_version: If True, skip ``jax.jit`` on the combined
                solve-and-simulate step (useful for debugging with Python-
                level tracebacks); the solve step itself is still whatever
                ``backward_induction`` does internally.

        Returns:
            A function ``solve_and_simulate_function(params) -> pd.DataFrame``
            with the same output as ``solve_and_simulate``.

        """
        # Fix everything except params, solution of the model and model_structure which contains large arrays.
        sim_func = lambda params, value, policy, endog_gid, model_structure: simulate_all_periods(
            states_initial=states_initial,
            n_periods=self.model_config["n_periods"],
            params=params,
            seed=seed,
            endog_grid_solved=endog_gid,
            policy_solved=policy,
            value_solved=value,
            model_config=self.model_config,
            model_structure=model_structure,
            model_funcs=self.model_funcs,
            alt_model_funcs_sim=self.alternative_sim_funcs,
        )

        (
            model_structure_for_jit,
            batch_info_for_jit,
            model_structure_non_jit,
            batch_info_non_jit,
        ) = split_structure_and_batch_info(self.model_structure, self.batch_info)

        def solve_and_simulate_function_to_jit(
            params, model_structure_jit, batch_info_jit
        ):
            params_processed = process_params(params, self.params_check_info)

            # Merge back parts together. The non_jit objects are fixed in the closure.
            model_structure = merge_non_jit_and_jit_model_structure(
                model_structure_jit, model_structure_non_jit
            )
            batch_info = merg_non_jit_batch_info_and_jit_batch_info(
                batch_info_jit, batch_info_non_jit
            )

            # Solve the model.
            value, policy, endog_grid = backward_induction(
                params=params_processed,
                model_structure=model_structure,
                batch_info=batch_info,
                income_shock_draws_unscaled=self.income_shock_draws_unscaled,
                income_shock_weights=self.income_shock_weights,
                model_config=self.model_config,
                model_funcs=self.model_funcs,
            )

            sim_dict = sim_func(
                params=params_processed,
                value=value,
                policy=policy,
                endog_gid=endog_grid,
                model_structure=model_structure,
            )

            return sim_dict

        if slow_version:
            solve_simulate_func = solve_and_simulate_function_to_jit
        else:
            solve_simulate_func = jax.jit(solve_and_simulate_function_to_jit)

        # Generate the function. The user only needs to provide params, but we call with the objects for jit.
        def solve_and_simulate_function(params: Dict[str, jnp.ndarray]) -> pd.DataFrame:
            sim_dict = solve_simulate_func(
                params, model_structure_for_jit, batch_info_for_jit
            )
            df = create_simulation_df(sim_dict)
            return df

        return solve_and_simulate_function

    def create_experimental_ll_func(
        self,
        params_all: Dict[str, float],
        observed_states: Dict[str, Any],
        observed_choices: jnp.ndarray,
        unobserved_state_specs: Optional[Dict[str, Any]] = None,
        return_model_solution: bool = False,
        use_probability_of_observed_states: bool = True,
        slow_version: bool = False,
    ) -> Callable[[Dict[str, float]], jnp.ndarray]:
        """Build a per-individual negative log-likelihood function for estimation.

        EXPERIMENTAL. Thin passthrough to
        ``create_individual_likelihood_function``; see that function's
        implementation for the exact choice-probability construction.

        Args:
            params_all: The full parameter dict; the function returned below
                is called with only the subset being estimated, and this dict
                supplies the rest.
            observed_states: Observed discrete (and continuous, if any) states
                from the data, one array per state name.
            observed_choices: Observed choices from the data.
            unobserved_state_specs: Optional specification of states that are
                unobserved in the data and must be integrated/weighted over
                when computing choice probabilities.
            return_model_solution: If True, the returned function also
                returns the ``(value, policy, endog_grid)`` solution dict
                alongside the likelihood contributions.
            use_probability_of_observed_states: Whether to weight by the
                model-implied probability of the observed (partially latent)
                states.
            slow_version: If True, skip ``jax.jit`` on the likelihood function
                (useful for debugging).

        Returns:
            A function mapping the estimated-parameter subset to per-
            individual negative log-likelihood contributions (and, if
            ``return_model_solution``, also the solution dict).

        """
        return create_individual_likelihood_function(
            income_shock_draws_unscaled=self.income_shock_draws_unscaled,
            income_shock_weights=self.income_shock_weights,
            batch_info=self.batch_info,
            model_structure=self.model_structure,
            model_config=self.model_config,
            model_funcs=self.model_funcs,
            model_specs=self.model_specs,
            observed_states=observed_states,
            observed_choices=observed_choices,
            params_all=params_all,
            unobserved_state_specs=unobserved_state_specs,
            return_model_solution=return_model_solution,
            use_probability_of_observed_states=use_probability_of_observed_states,
            slow_version=slow_version,
        )

    def validate_exogenous(self, params: Dict[str, jnp.ndarray]) -> bool:
        """Check that every stochastic transition is a valid probability vector.

        Verifies, for each exogenous (stochastic) process and every state it
        is evaluated at, that the transition probabilities are non-negative,
        sum to one, and have the expected dimensionality.

        Args:
            params: Model parameters, in any form ``process_params`` accepts.

        Returns:
            True if every exogenous process validates; otherwise a
            ``ValueError`` is raised by ``validate_stochastic_transition``.

        """
        return validate_stochastic_transition(
            params=params,
            model_structure=self.model_structure,
            model_config=self.model_config,
            model_funcs=self.model_funcs,
        )

    def get_state_choices_idx(self, states: Dict[str, Any]) -> jnp.ndarray:
        """Look up state-choice indices for given discrete states.

        Args:
            states: Discrete state values, one array (or scalar) per state
                name.

        Returns:
            The index of each state's block in the state-choice space (i.e.
            the row at which its first choice starts).

        """
        return get_state_choice_index_per_discrete_states(
            states=states,
            map_state_choice_to_index=self.model_structure["map_state_choice_to_index"],
            discrete_states_names=self.model_structure["discrete_states_names"],
        )

    def get_child_states(self, state: Dict[str, Any], choice: Any) -> pd.DataFrame:
        """Look up the (deterministic) child states for a state-choice.

        Requires the model to have been built with ``debug_info="all"``, since
        ``map_state_choice_to_child_states`` is only kept around in that mode.

        Args:
            state: A discrete state (or a batch of them).
            choice: The choice taken in ``state``.

        Returns:
            A DataFrame with one row per child state, one column per discrete
            state name.

        """
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

    def get_child_states_and_calc_trans_probs(
        self, state: Dict[str, Any], choice: Any, params: Dict[str, jnp.ndarray]
    ) -> pd.DataFrame:
        """``get_child_states`` plus each child's stochastic transition probability.

        Args:
            state: A discrete state (or a batch of them).
            choice: The choice taken in ``state``.
            params: Model parameters, in any form ``process_params`` accepts.

        Returns:
            The ``get_child_states`` DataFrame with an added ``trans_probs``
            column.

        """
        child_states_df = self.get_child_states(state, choice)

        trans_probs = self.model_funcs["compute_stochastic_transition_vec"](
            params=params, choice=choice, **state
        )
        child_states_df["trans_probs"] = trans_probs
        return child_states_df

    def get_full_child_states_by_asset_id_and_probs(
        self,
        state: Dict[str, Any],
        choice: Any,
        params: Dict[str, jnp.ndarray],
        asset_id: int,
        second_continuous_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """``get_child_states_and_calc_trans_probs`` plus next-period wealth by
        quadrature point.

        Requires the model to have been built with ``debug_info="all"``.

        Args:
            state: A discrete state (or a batch of them).
            choice: The choice taken in ``state``.
            params: Model parameters, in any form ``process_params`` accepts.
            asset_id: Index into the end-of-period asset grid to evaluate the
                law of motion at.
            second_continuous_id: Index into the additional continuous-state
                grid, required if and only if the model has one.

        Returns:
            The ``get_child_states_and_calc_trans_probs`` DataFrame, with an
            added ``assets_begin_of_period_quad_point_{i}`` column per income
            quadrature point (and, with an additional continuous state, an
            added column for its next-period value).

        """
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

        continuous_states_info = self.model_config["continuous_states_info"]

        if continuous_states_info["has_additional_continuous_state"]:
            if second_continuous_id is None:
                raise ValueError("second_continuous_id must be provided.")
            else:
                quad_wealth = child_continuous_states["assets_begin_of_period"][
                    child_idx, second_continuous_id, asset_id, :
                ]

                for continuous_state_name in continuous_states_info[
                    "additional_continuous_state_names"
                ]:
                    next_period_continuous_state = child_continuous_states[
                        "continuous_states"
                    ][continuous_state_name][child_idx, second_continuous_id]

                    child_states_df[continuous_state_name] = (
                        next_period_continuous_state
                    )

        else:
            if second_continuous_id is not None:
                raise ValueError("second_continuous_id must not be provided.")
            else:
                # Wealth-only models carry a size-1 dummy continuous dimension
                quad_wealth = child_continuous_states["assets_begin_of_period"][
                    child_idx, 0, asset_id, :
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

    def compute_law_of_motions(self, params: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """Compute the law of motion for the whole state(-choice) space at once.

        Debug/inspection entry point: unlike the main solve path (which
        computes this on demand, per batch and period), this evaluates
        ``calc_cont_grids_next_period`` once for every state-choice, which is
        only tractable for small-to-medium models.

        Args:
            params: Model parameters, in any form ``process_params`` accepts.

        Returns:
            A dict with ``"assets_begin_of_period"`` and ``"continuous_states"``
            arrays, indexed by the full state(-choice) space.

        """
        return calc_cont_grids_next_period(
            params=params,
            model_structure=self.model_structure,
            model_config=self.model_config,
            model_funcs=self.model_funcs,
            income_shock_draws_unscaled=self.income_shock_draws_unscaled,
        )

    def get_n_state_choices_per_period(self) -> pd.Series:
        """Count state-choices per period.

        Returns:
            A pandas Series indexed by period, with the number of
            state-choices in each.

        """
        return get_n_state_choice_period(self.model_structure)

    def solve_partially(
        self,
        params: Dict[str, jnp.ndarray],
        n_periods: int,
        return_candidates: bool = False,
    ) -> Dict[str, Any]:
        """Solve only the last ``n_periods`` periods, for debugging large models.

        Args:
            params: Model parameters, in any form ``process_params`` accepts.
            n_periods: Number of periods (counted from the end) to solve.
            return_candidates: If True, also return the pre-upper-envelope
                candidate solutions.

        Returns:
            A dict with the solved ``value``/``policy``/``endog_grid`` (and,
            if ``return_candidates``, the candidate solutions) for the solved
            periods.

        """
        return partially_solve(
            income_shock_draws_unscaled=self.income_shock_draws_unscaled,
            income_shock_weights=self.income_shock_weights,
            model_config=self.model_config,
            batch_info=self.batch_info,
            model_funcs=self.model_funcs,
            model_structure=self.model_structure,
            params=params,
            n_periods=n_periods,
            return_candidates=return_candidates,
        )

    def set_alternative_sim_funcs(
        self,
        alternative_sim_specifications: Dict[str, Callable],
        alternative_specs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """(Re-)build ``self.alternative_sim_funcs`` for use in simulation.

        Args:
            alternative_sim_specifications: Alternative simulation functions,
                passed through to ``generate_alternative_sim_functions``.
            alternative_specs: Optional alternative model specs to build them
                against; defaults to this model's own ``model_specs`` if
                omitted.

        """
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
