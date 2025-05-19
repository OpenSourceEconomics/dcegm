from functools import partial
from typing import Callable, Dict

import jax

from dcegm.interfaces.sol_interface import model_solved
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.check_params import process_params
from dcegm.pre_processing.setup_model import create_model_dict
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
        model_dict = create_model_dict(
            model_config=model_config,
            model_specs=model_specs,
            utility_functions=utility_functions,
            utility_functions_final_period=utility_functions_final_period,
            budget_constraint=budget_constraint,
            state_space_functions=state_space_functions,
            exogenous_states_transition=exogenous_states_transition,
            shock_functions=shock_functions,
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

    def solve(self, params):
        """
        Solve the model using backward induction.

        Args:
            params: The parameters for the model.

        Returns:
            A dictionary containing the solution of the model.
        """
        params_processed = process_params(params)
        # Solve the model
        value, policy, endog_grid = self.backward_induction_jit(params_processed)

        model_config = self.model_config
        model_structure = self.model_structure
        model_funcs = self.model_funcs

        model_solved_class = model_solved(
            value=value,
            policy=policy,
            endog_grid=endog_grid,
            model_config=model_config,
            model_structure=model_structure,
            model_funcs=model_funcs,
            params=params_processed,
        )
        return model_solved_class
