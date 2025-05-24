import jax.numpy as jnp

from dcegm.interfaces.interface import (
    policy_and_value_for_state_choice_vec,
    policy_for_state_choice_vec,
    value_for_state_choice_vec,
)
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods


class model_solved:
    def __init__(
        self,
        value,
        policy,
        endog_grid,
        model_config,
        model_structure,
        model_funcs,
        params,
        alternative_sim_funcs,
    ):
        self.value = value
        self.policy = policy
        self.endog_grid = endog_grid
        self.model_config = model_config
        self.model_structure = model_structure
        self.model_funcs = model_funcs
        self.params = params
        self.alternative_sim_funcs = alternative_sim_funcs

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

    def value_and_policy_for_state_and_choice(self, state, choice):
        """
        Get the value and policy for a given state and choice.

        Args:
            state: The state for which to get the value and policy.
            choice: The choice for which to get the value and policy.

        Returns:
            A tuple containing the value and policy for the given state and choice.
        """
        return policy_and_value_for_state_choice_vec(
            states=state,
            choice=choice,
            model_config=self.model_config,
            model_structure=self.model_structure,
            model_funcs=self.model_funcs,
            params=self.params,
            endog_grid_solved=self.endog_grid,
            value_solved=self.value,
            policy_solved=self.policy,
        )

    def value_for_state_and_choice(self, state, choice):
        """
        Get the value for a given state and choice.

        Args:
            state: The state for which to get the value.
            choice: The choice for which to get the value.

        Returns:
            The value for the given state and choice.
        """

        return value_for_state_choice_vec(
            states=state,
            choice=choice,
            model_config=self.model_config,
            model_structure=self.model_structure,
            model_funcs=self.model_funcs,
            params=self.params,
            endog_grid_solved=self.endog_grid,
            value_solved=self.value,
        )

    def policy_for_state_and_choice(self, state, choice):
        """Get the policy for a given state and choice.

        Args:
            state: The state for which to get the policy.
            choice: The choice for which to get the policy.

        Returns:
            The policy for the given state and choice.

        """

        return policy_for_state_choice_vec(
            states=state,
            choice=choice,
            model_config=self.model_config,
            model_structure=self.model_structure,
            endog_grid_solved=self.endog_grid,
            policy_solved=self.policy,
        )

    def get_solution_for_discrete_state_choice(self, states, choice):
        """Get the solution container for a given discrete state and choice combination.

        Args:
            states: The state for which to get the solution.
            choice: The choice for which to get the solution.
        Returns:
            A tuple containing the wealth grid, value grid, and policy grid for the given state and choice.

        """
        # Get the value and policy for a given state and choice.

        map_state_choice_to_index = self.model_structure[
            "map_state_choice_to_index_with_proxy"
        ]
        discrete_states_names = self.model_structure["discrete_states_names"]

        if "dummy_stochastic" in discrete_states_names:
            state_choice_vec = {
                **states,
                "choice": choice,
                "dummy_stochastic": 0,
            }
        else:
            state_choice_vec = {
                **states,
                "choice": choice,
            }

        state_choice_tuple = tuple(
            state_choice_vec[st] for st in discrete_states_names + ["choice"]
        )
        state_choice_index = map_state_choice_to_index[state_choice_tuple]

        wealth_grid = jnp.take(self.endog_grid, state_choice_index, axis=0)
        value_grid = jnp.take(self.value, state_choice_index, axis=0)
        policy_grid = jnp.take(self.policy, state_choice_index, axis=0)

        # find the nonNaN index
        non_nan_index = jnp.where(jnp.isfinite(wealth_grid))[0]

        # return the nonNaN values
        if non_nan_index.size > 0:
            wealth_grid = wealth_grid[non_nan_index]
            value_grid = value_grid[non_nan_index]
            policy_grid = policy_grid[non_nan_index]
        else:
            # If all values are NaN, return empty arrays
            wealth_grid = jnp.array([])
            value_grid = jnp.array([])
            policy_grid = jnp.array([])

        return wealth_grid, value_grid, policy_grid
