import jax.numpy as jnp

from dcegm.interfaces.index_functions import (
    get_state_choice_index_per_discrete_states_and_choices,
)
from dcegm.interfaces.interface import (
    policy_and_value_for_states_and_choices,
    policy_for_state_choice_vec,
    value_for_state_and_choice,
)
from dcegm.interfaces.interface_checks import check_states_and_choices
from dcegm.likelihood import (
    calc_choice_probs_for_states,
    choice_values_for_states,
    get_state_choice_index_per_discrete_states,
)
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods


class model_solved:
    def __init__(
        self,
        model,
        params,
        value,
        policy,
        endog_grid,
    ):
        # Assign the solution containers and params
        self.params = params
        self.value = value
        self.policy = policy
        self.endog_grid = endog_grid

        # Assign the model itself
        self.model = model

        # Assign all attributes from the model
        self.model_config = model.model_config
        self.model_structure = model.model_structure
        self.model_funcs = model.model_funcs
        self.model_specs = model.model_specs
        self.alternative_sim_funcs = model.alternative_sim_funcs

    def simulate(self, states_initial, seed):

        sim_dict = simulate_all_periods(
            states_initial=states_initial,
            n_periods=self.model_config["n_periods"],
            params=self.params,
            seed=seed,
            endog_grid_solved=self.endog_grid,
            policy_solved=self.policy,
            value_solved=self.value,
            model_config=self.model_config,
            model_structure=self.model_structure,
            model_funcs=self.model_funcs,
            alt_model_funcs_sim=self.alternative_sim_funcs,
        )
        return create_simulation_df(sim_dict)

    def value_and_policy_for_states_and_choices(self, states, choices):
        """Get the value and policy for a given state and choice.

        Args:
            states: The state for which to get the value and policy.
            choices: The choice for which to get the value and policy.

        Returns:
            A tuple containing the value and policy for the given state and choice.

        """
        return policy_and_value_for_states_and_choices(
            states=states,
            choices=choices,
            model_config=self.model_config,
            model_structure=self.model_structure,
            model_funcs=self.model_funcs,
            params=self.params,
            endog_grid_solved=self.endog_grid,
            value_solved=self.value,
            policy_solved=self.policy,
        )

    def value_for_states_and_choices(self, states, choices):
        """Get the value for a given state and choice.

        Args:
            states: The state for which to get the value.
            choices: The choice for which to get the value.

        Returns:
            The value for the given state and choice.

        """

        return value_for_state_and_choice(
            states=states,
            choices=choices,
            model_config=self.model_config,
            model_structure=self.model_structure,
            model_funcs=self.model_funcs,
            params=self.params,
            endog_grid_solved=self.endog_grid,
            value_solved=self.value,
        )

    def policy_for_states_and_choices(self, states, choices):
        """Get the policy for a given state and choice.

        Args:
            states: The state for which to get the policy.
            choices: The choice for which to get the policy.

        Returns:
            The policy for the given state and choice.

        """

        return policy_for_state_choice_vec(
            states=states,
            choices=choices,
            model_config=self.model_config,
            model_structure=self.model_structure,
            endog_grid_solved=self.endog_grid,
            policy_solved=self.policy,
        )

    def get_solution_for_discrete_state_choice(self, states, choices):
        """Get the solution container for a given discrete state and choice combination.

        Args:
            states: The state for which to get the solution.
            choices: The choice for which to get the solution.
        Returns:
            A tuple containing the wealth grid, value grid, and policy grid for the given state and choice.

        """
        # Check if the states and choices are valid according to the model structure.
        state_choices = check_states_and_choices(
            states=states,
            choices=choices,
            model_structure=self.model_structure,
        )

        # Get the value and policy for a given state and choice. We use state choices as states as it is not important
        # that these are missing.
        state_choice_index = get_state_choice_index_per_discrete_states_and_choices(
            model_structure=self.model_structure,
            states=state_choices,
            choices=state_choices["choice"],
        )

        endog_grid = jnp.take(self.endog_grid, state_choice_index, axis=0)
        value_grid = jnp.take(self.value, state_choice_index, axis=0)
        policy_grid = jnp.take(self.policy, state_choice_index, axis=0)

        return endog_grid, value_grid, policy_grid

    def choice_probabilities_for_states(self, states):

        state_choice_idxs = get_state_choice_index_per_discrete_states(
            states=states,
            map_state_choice_to_index=self.model_structure[
                "map_state_choice_to_index_with_proxy"
            ],
            discrete_states_names=self.model_structure["discrete_states_names"],
        )

        return calc_choice_probs_for_states(
            value_solved=self.value,
            endog_grid_solved=self.endog_grid,
            state_choice_indexes=state_choice_idxs,
            params=self.params,
            states=states,
            model_config=self.model_config,
            model_funcs=self.model_funcs,
        )

    def choice_values_for_states(self, states):
        state_choice_idxs = get_state_choice_index_per_discrete_states(
            states=states,
            map_state_choice_to_index=self.model_structure[
                "map_state_choice_to_index_with_proxy"
            ],
            discrete_states_names=self.model_structure["discrete_states_names"],
        )
        return choice_values_for_states(
            value_solved=self.value,
            endog_grid_solved=self.endog_grid,
            state_choice_indexes=state_choice_idxs,
            params=self.params,
            states=states,
            model_config=self.model_config,
            model_funcs=self.model_funcs,
        )
