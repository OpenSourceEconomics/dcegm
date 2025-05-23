import jax.numpy as jnp
import numpy as np


def check_model_config_and_process(model_config):
    """Check if options are valid and set defaults."""

    processed_model_config = {}

    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary.")

    if "n_periods" not in model_config:
        raise ValueError("model_config must contain n_periods.")

    if not isinstance(model_config["n_periods"], int):
        raise ValueError("Number of periods must be an integer.")

    if not model_config["n_periods"] > 1:
        raise ValueError("Number of periods must be greater than 1.")

    if "n_quad_points" not in model_config:
        raise ValueError("model_config must contain n_quad_points.")

    if not isinstance(model_config["n_quad_points"], int):
        raise ValueError("Number of quadrature points must be an integer.")

    processed_model_config["n_periods"] = model_config["n_periods"]
    processed_model_config["n_quad_points"] = model_config["n_quad_points"]

    # This checks if choices is a list or an integer
    if "choices" in model_config:
        if isinstance(model_config["choices"], list):
            processed_model_config["choices"] = np.array(
                model_config["choices"], dtype=np.uint8
            )
        elif isinstance(model_config["choices"], int):
            processed_model_config["choices"] = np.array(
                [model_config["choices"]], dtype=np.uint8
            )
        elif isinstance(model_config["choices"], np.ndarray):
            processed_model_config["choices"] = model_config["choices"].astype(np.uint8)
        else:
            raise ValueError("Choices must be a list or an integer.")

    else:
        raise ValueError("Choices must be given in model_config.")

    if "continuous_states" not in model_config:
        raise ValueError("model_config must contain continuous_states as key.")

    continuous_states_grids = model_config["continuous_states"].copy()

    if not isinstance(continuous_states_grids, dict):
        raise ValueError("model_config['continuous_states'] must be a dictionary.")

    if "assets_end_of_period" not in continuous_states_grids:
        raise ValueError(
            "model_config['assets_end_of_period'] must contain wealth as key."
        )
    # Check if it is an array
    asset_grid = continuous_states_grids["assets_end_of_period"]
    if not isinstance(asset_grid, (list, np.ndarray, jnp.ndarray)):
        raise ValueError(
            "model_config['continuous_states']['assets_end_of_period'] must be a list or an array."
        )

    # ToDo: Check if it is monotonic increasing

    continuous_states_info = {}
    n_assets_end_of_period = len(asset_grid)
    continuous_states_info["assets_grid_end_of_period"] = jnp.asarray(
        continuous_states_grids["assets_end_of_period"], dtype=float
    )

    if len(continuous_states_grids) > 2:
        raise ValueError("At most two continuous states are supported.")

    elif len(continuous_states_grids) == 2:
        second_continuous_state = next(
            (
                {key: value}
                for key, value in model_config["continuous_states"].items()
                if key != "assets_end_of_period"
            ),
            None,
        )

        continuous_states_info["second_continuous_exists"] = True

        second_continuous_state_name = list(second_continuous_state.keys())[0]
        continuous_states_info["second_continuous_state_name"] = (
            second_continuous_state_name
        )

        second_continuous_state_grid = continuous_states_grids[
            second_continuous_state_name
        ]
        continuous_states_info["second_continuous_grid"] = second_continuous_state_grid
        # ToDo: Check if grid is array or list and monotonic increasing

        continuous_states_info["n_second_continuous_grid"] = len(
            second_continuous_state_grid
        )

    else:
        continuous_states_info["second_continuous_exists"] = False
        continuous_states_info["second_continuous_state_name"] = None
        continuous_states_info["n_second_continuous_grid"] = None
        continuous_states_info["second_continuous_grid"] = None

    processed_model_config["continuous_states_info"] = continuous_states_info

    if "tuning_params" not in model_config:
        tuning_params = {}
    else:
        tuning_params = model_config["tuning_params"]

    tuning_params["extra_wealth_grid_factor"] = (
        tuning_params["extra_wealth_grid_factor"]
        if "extra_wealth_grid_factor" in tuning_params
        else 0.2
    )
    tuning_params["n_constrained_points_to_add"] = (
        tuning_params["n_constrained_points_to_add"]
        if "n_constrained_points_to_add" in tuning_params
        else n_assets_end_of_period // 10
    )

    if (
        n_assets_end_of_period * (1 + tuning_params["extra_wealth_grid_factor"])
        < n_assets_end_of_period + tuning_params["n_constrained_points_to_add"]
    ):
        raise ValueError(
            f"""\n\n
            When preparing the tuning parameters for the upper
            envelope, we found the following contradicting parameters: \n
            The extra wealth grid factor of {tuning_params["extra_wealth_grid_factor"]} is too small
            to cover the {tuning_params["n_constrained_points_to_add"]} wealth points which are added in
            the credit constrained part of the wealth grid. \n\n"""
        )
    tuning_params["n_total_wealth_grid"] = int(
        n_assets_end_of_period * (1 + tuning_params["extra_wealth_grid_factor"])
    )

    # Set jump threshold to default 2 if it is not given
    tuning_params["fues_jump_thresh"] = int(
        tuning_params["fues_jump_threshold"]
        if "fues_jump_threshold" in tuning_params
        else 2
    )

    # Set fues_n_points_to_scan to 10 if not given
    tuning_params["fues_n_points_to_scan"] = int(
        tuning_params["fues_n_points_to_scan"]
        if "fues_n_points_to_scan" in tuning_params
        else 10
    )

    processed_model_config["tuning_params"] = tuning_params

    if "min_period_batch_segments" in model_config.keys():
        processed_model_config["min_period_batch_segments"] = model_config[
            "min_period_batch_segments"
        ]
    else:
        processed_model_config["min_period_batch_segments"] = None

    if "stochastic_states" in model_config.keys():
        processed_model_config["stochastic_states"] = model_config["stochastic_states"]

    if "deterministic_states" in model_config.keys():
        processed_model_config["deterministic_states"] = model_config[
            "deterministic_states"
        ]

    processed_model_config["params_check_info"] = {}

    return processed_model_config
