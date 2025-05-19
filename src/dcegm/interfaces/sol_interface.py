from dcegm.interfaces.interface import policy_and_value_for_state_choice_vec


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
    ):
        self.value = value
        self.policy = policy
        self.endog_grid = endog_grid
        self.model_config = model_config
        self.model_structure = model_structure
        self.model_funcs = model_funcs
        self.params = params

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
