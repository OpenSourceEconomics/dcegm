import pickle

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from upper_envelope import fues_jax

from dcegm.upper_envelope_fedor import upper_envelope


def create_upper_envelope_function(model_config, continuous_state=None):
    if len(model_config["choices"]) < 2:
        compute_upper_envelope = no_upper_envelope_dummy_function
    else:

        tuning_params = model_config["tuning_params"]

        if continuous_state:

            def compute_upper_envelope(
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

        else:

            def compute_upper_envelope(
                endog_grid,
                policy,
                value,
                expected_value_zero_assets,
                state_choice_dict,
                utility_function,
                params,
                discount_factor,
            ):
                policy_two_dim = np.vstack((endog_grid, policy))
                value_two_dim = np.vstack((endog_grid, value))

                params["beta"] = discount_factor

                # breakpoint()  # Set a breakpoint here to inspect the inputs

                choice = state_choice_dict["choice"]
                lagged_choice = state_choice_dict["lagged_choice"]
                period = state_choice_dict["period"]

                label = f"choice{choice}_lagged{lagged_choice}_period{period}"

                with open(f"pickle/pre_ue_policy_{label}.pickle", "wb") as f:
                    pickle.dump(policy, f)

                with open(f"pickle/pre_ue_value_{label}.pickle", "wb") as f:
                    pickle.dump(value, f)

                with open(f"pickle/pre_ue_endog_grid_{label}.pickle", "wb") as f:
                    pickle.dump(endog_grid, f)

                policy_sol, value_sol = upper_envelope(
                    policy=policy_two_dim,
                    value=value_two_dim,
                    state_choice_vec=state_choice_dict,
                    params=params,
                    compute_utility=utility_function,
                )
                n_nans_policy = (np.isnan(policy_sol[0, :])).sum()
                n_nans_value = (np.isnan(value_sol[0, :])).sum()

                endog_grid_fedor = policy_sol[0, :]
                policy_fedor = policy_sol[1, :]
                value_fedor = value_sol[1, :]
                endog_grid_fedor = np.append(0.0, endog_grid_fedor)
                policy_fedor = np.append(0.0, policy_fedor)
                value_fedor = np.append(expected_value_zero_assets, value_fedor)

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

                endog_grid_fues, policy_fues, value_fues = fues_jax(
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
                not_nan_fedor = np.isfinite(endog_grid_fedor)
                not_nan_fues = np.isfinite(endog_grid_fues)

                # print value fedor and fues index 13 and 14
                print(
                    f"fedor: {endog_grid_fedor[13:17]}, {value_fedor[13:17]}, fues: {endog_grid_fues[13:17]}, {value_fues[13:17]}"
                )

                # now just the difference between the two
                print(
                    f"fedor - fues: {endog_grid_fedor[13:17] - endog_grid_fues[13:17]}, {value_fedor[13:17] - value_fues[13:17]}"
                )

                # remove values at index 15 from fues
                endog_grid_fues = np.delete(endog_grid_fues, 15)
                policy_fues = np.delete(policy_fues, 15)
                value_fues = np.delete(value_fues, 15)
                endog_grid_fedor[0:100] - endog_grid_fues[0:100]
                policy_fedor[0:100] - policy_fues[0:100]
                value_fedor[0:100] - value_fues[0:100]

                assert np.allclose(
                    endog_grid_fedor[not_nan_fedor], endog_grid_fues[not_nan_fues]
                )
                assert np.allclose(
                    policy_fedor[not_nan_fedor], policy_fues[not_nan_fues]
                )
                assert np.allclose(value_fedor[not_nan_fedor], value_fues[not_nan_fues])
                return endog_grid_fedor, policy_fedor, value_fedor

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
