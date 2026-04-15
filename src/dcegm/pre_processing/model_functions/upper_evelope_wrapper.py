from jax import numpy as jnp
from upper_envelope.jax import drued_jorg_jax, fues_jax


def create_upper_envelope_function(model_config, continuous_state=None):
    if len(model_config["choices"]) < 2:
        compute_upper_envelope = no_upper_envelope_dummy_function
    else:

        tuning_params = model_config["upper_envelope"]["tuning_params"]

        if continuous_state:

            def compute_upper_envelope(
                model_config,
                endog_grid,
                policy,
                value,
                expected_value_zero_assets,
                second_continuous_state,
                state_choice_dict,
                utility_function,
                params,
                discount_factor,
            ):
                value_kwargs = {
                    "second_continuous_state": second_continuous_state,
                    "expected_value_zero_assets": expected_value_zero_assets,
                    "params": params,
                    "discount_factor": discount_factor,
                    **state_choice_dict,
                }

                def value_function(
                    consumption,
                    second_continuous_state,
                    expected_value_zero_assets,
                    params,
                    discount_factor,
                    **state_choice_dict,
                ):
                    return (
                        utility_function(
                            consumption=consumption,
                            continuous_state=second_continuous_state,
                            params=params,
                            **state_choice_dict,
                        )
                        + discount_factor * expected_value_zero_assets
                    )

                if model_config["upper_envelope"]["method"] == "fues":
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

                elif model_config["upper_envelope"]["method"] == "drued_jorg":
                    return drued_jorg_jax(
                        endog_grid=endog_grid,
                        policy=policy,
                        value=value,
                        expected_value_zero_savings=expected_value_zero_assets,
                        value_function=value_function,
                        value_function_kwargs=value_kwargs,
                        m_grid=tuning_params["m_grid"],
                    )
                else:
                    raise ValueError(
                        f"Unknown upper envelope method: {model_config['upper_envelope_method']}. Choose 'fues' or 'drued_jorg'."
                    )

        else:

            def compute_upper_envelope(
                model_config,
                endog_grid,
                policy,
                value,
                expected_value_zero_assets,
                state_choice_dict,
                utility_function,
                params,
                discount_factor,
            ):
                value_kwargs = {
                    "expected_value_zero_assets": expected_value_zero_assets,
                    "params": params,
                    "discount_factor": discount_factor,
                    **state_choice_dict,
                }

                def value_function(
                    consumption,
                    expected_value_zero_assets,
                    params,
                    discount_factor,
                    **state_choice_dict,
                ):
                    return (
                        utility_function(
                            consumption=consumption, params=params, **state_choice_dict
                        )
                        + discount_factor * expected_value_zero_assets
                    )

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

    return compute_upper_envelope


def no_upper_envelope_dummy_function(
    model_config, endog_grid, policy, value, expected_value_zero_savings, *args
):
    """This is a dummy function for the case of only one discrete choice."""
    n_nans = int(0.2 * endog_grid.shape[0])

    nans_to_append = jnp.full(n_nans - 1, jnp.nan)
    endog_grid = jnp.append(jnp.append(0, endog_grid), nans_to_append)
    policy = jnp.append(jnp.append(0, policy), nans_to_append)
    value = jnp.append(jnp.append(expected_value_zero_savings, value), nans_to_append)

    return endog_grid, policy, value
