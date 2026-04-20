from jax import numpy as jnp
from upper_envelope.jax import drued_jorg_jax, fues_jax


def create_upper_envelope_function(
    model_config,
    continuous_state=None,
    has_additional_continuous_states=None,
):
    if len(model_config["choices"]) < 2:
        return no_upper_envelope_dummy_function

    if has_additional_continuous_states is None:
        has_additional_continuous_states = bool(continuous_state)

    tuning_params = model_config["upper_envelope"]["tuning_params"]
    method = model_config["upper_envelope"]["method"]

    def compute_upper_envelope(
        endog_grid,
        policy,
        value,
        expected_value_zero_assets,
        continuous_state_dict,
        state_choice_dict,
        utility_function,
        params,
        discount_factor,
    ):
        continuous_state_dict = (
            {} if continuous_state_dict is None else continuous_state_dict
        )
        state_choice_vars = {**state_choice_dict, **continuous_state_dict}

        value_kwargs = {
            "expected_value_zero_assets": expected_value_zero_assets,
            "params": params,
            "discount_factor": discount_factor,
        }

        def value_function(
            consumption,
            expected_value_zero_assets,
            params,
            discount_factor,
        ):
            return (
                utility_function(
                    consumption=consumption,
                    params=params,
                    **state_choice_vars,
                )
                + discount_factor * expected_value_zero_assets
            )

        # --- method dispatch ---
        if method == "fues":
            return fues_jax(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                expected_value_zero_savings=expected_value_zero_assets,
                value_function=value_function,
                value_function_kwargs=value_kwargs,
                n_constrained_points_to_add=tuning_params[
                    "n_constrained_points_to_add"
                ],
                n_final_wealth_grid=tuning_params["n_total_wealth_grid"],
                jump_thresh=tuning_params["fues_jump_thresh"],
                n_points_to_scan=tuning_params["fues_n_points_to_scan"],
            )

        elif method == "druedahl_jorgensen":
            return drued_jorg_jax(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                expected_value_zero_savings=expected_value_zero_assets,
                value_function=value_function,
                value_function_kwargs=value_kwargs,
                m_grid=model_config["continuous_states_info"]["assets_begin_of_period"],
            )

        else:
            raise ValueError(
                f"Unknown upper envelope method: {method}. "
                "Choose 'fues' or 'druedahl_jorgensen'."
            )

    return compute_upper_envelope


def no_upper_envelope_dummy_function(
    endog_grid, policy, value, expected_value_zero_savings, *args
):
    """This is a dummy function for the case of only one discrete choice."""
    n_nans = int(0.2 * endog_grid.shape[0])

    nans_to_append = jnp.full(n_nans - 1, jnp.nan)
    endog_grid = jnp.append(jnp.append(0, endog_grid), nans_to_append)
    policy = jnp.append(jnp.append(0, policy), nans_to_append)
    value = jnp.append(jnp.append(expected_value_zero_savings, value), nans_to_append)

    return endog_grid, policy, value
