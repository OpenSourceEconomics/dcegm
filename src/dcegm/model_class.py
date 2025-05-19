import pickle
from functools import partial
from typing import Callable, Dict

import jax

from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.batches.batch_creation import create_batches_and_information
from dcegm.pre_processing.check_options import check_model_config_and_process
from dcegm.pre_processing.check_params import process_params
from dcegm.pre_processing.model_functions.process_model_functions import (
    process_model_functions,
)
from dcegm.pre_processing.model_structure.exogenous_processes import (
    create_exog_state_mapping,
)
from dcegm.pre_processing.model_structure.model_structure import create_model_structure
from dcegm.pre_processing.shared import create_array_with_smallest_int_dtype
from dcegm.solve import backward_induction


class setup_model:
    def __init__(
        self,
        model_config: Dict,
        model_specs: Dict,
        utility_functions: Dict[str, Callable],
        utility_functions_final_period: Dict[str, Callable],
        budget_constraint: Callable,
        state_space_functions: Dict[str, Callable] = None,
        exogenous_states_transition: Dict[str, Callable] = None,
        shock_functions: Dict[str, Callable] = None,
    ):
        model_config = check_model_config_and_process(model_config)

        model_funcs = process_model_functions(
            model_config=model_config,
            model_specs=model_specs,
            state_space_functions=state_space_functions,
            utility_functions=utility_functions,
            utility_functions_final_period=utility_functions_final_period,
            budget_constraint=budget_constraint,
            exogenous_states_transitions=exogenous_states_transition,
            shock_functions=shock_functions,
        )

        model_structure = create_model_structure(
            model_config=model_config,
            model_funcs=model_funcs,
        )

        model_funcs["exog_state_mapping"] = create_exog_state_mapping(
            model_structure["exog_state_space"],
            model_structure["exog_states_names"],
        )

        print("State, state-choice and child state mapping created.\n")
        print("Start creating batches for the model.")

        batch_info = create_batches_and_information(
            model_structure=model_structure,
            n_periods=model_config["n_periods"],
            min_period_batch_segments=model_config["min_period_batch_segments"],
        )
        # Delete large array which is not needed. Not if all is requested
        # by the debug string.
        model_structure.pop("map_state_choice_to_child_states")
        print("Model setup complete.\n")
        self.model_config = model_config
        self.model_funcs = model_funcs
        self.model_structure = model_structure
        self.batch_info = jax.tree.map(create_array_with_smallest_int_dtype, batch_info)

        income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
            model_config["n_quad_points"]
        )

        self.income_shock_draws_unscaled = income_shock_draws_unscaled
        self.income_shock_weights = income_shock_weights

        has_second_continuous_state = model_config["continuous_states_info"][
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

    def solve(self, params):
        """
        Solve the model using backward induction.

        Args:
            params: The parameters for the model.

        Returns:
            A dictionary containing the solution of the model.
        """
        params_initial = process_params(params)
        # Solve the model
        solution = self.backward_induction_jit(params_initial)
        return solution
