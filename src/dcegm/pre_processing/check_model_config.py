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

    additional_continuous_states = {
        key: value
        for key, value in continuous_states_grids.items()
        if key not in ("assets_end_of_period", "assets_begin_of_period")
    }

    continuous_states_info["additional_continuous_state_names"] = list(
        additional_continuous_states.keys()
    )
    continuous_states_info["additional_continuous_state_grids"] = {
        key: jnp.asarray(value) for key, value in additional_continuous_states.items()
    }
    continuous_states_info["n_additional_continuous_states"] = len(
        additional_continuous_states
    )
    continuous_states_info["has_additional_continuous_state"] = (
        continuous_states_info["n_additional_continuous_states"] > 0
    )

    processed_model_config["continuous_states_info"] = continuous_states_info

    # Set default upper envelope method if not given.
    if "upper_envelope" not in model_config:
        upper_envelope = {}
        upper_envelope["method"] = "fues"
    elif "method" not in model_config["upper_envelope"]:
        upper_envelope = model_config["upper_envelope"]
        upper_envelope["method"] = "fues"
    elif (
        "upper_envelope" in model_config
        and model_config["upper_envelope"]["method"] == "druedahl_jorgensen"
        and "tuning_params" in model_config["upper_envelope"]
    ):
        raise ValueError(
            "'tuning_params' cannot be used with the 'druedahl_jorgensen',"
            " specify 'begin_of_period_assets_grid' in 'continuous_states' instead and delete 'tuning_params' "
            "from the model_config['upper_envelope']"
        )
    else:
        upper_envelope = dict(model_config["upper_envelope"])

    if "tuning_params" not in upper_envelope:
        tuning_params = {}
    elif "tuning_params" in model_config:
        raise ValueError(
            "tuning_params should be nested in model_config['upper_envelope']"
        )
    else:
        tuning_params = model_config["upper_envelope"]["tuning_params"]

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
        raise ValueError(f"""\n\n
            When preparing the tuning parameters for the upper
            envelope, we found the following contradicting parameters: \n
            The extra wealth grid factor of {tuning_params["extra_wealth_grid_factor"]} is too small
            to cover the {tuning_params["n_constrained_points_to_add"]} wealth points which are added in
            the credit constrained part of the wealth grid. \n\n""")
    tuning_params["n_total_wealth_grid"] = int(
        n_assets_end_of_period * (1 + tuning_params["extra_wealth_grid_factor"])
    )

    # Set jump threshold to default 2 if it is not given
    tuning_params["fues_jump_thresh"] = int(
        tuning_params["fues_jump_threshold"]
        if ("fues_jump_threshold" in tuning_params)
        & (upper_envelope["method"] == "fues")
        else 2
    )

    # Set fues_n_points_to_scan to 10 if not given
    tuning_params["fues_n_points_to_scan"] = int(
        tuning_params["fues_n_points_to_scan"]
        if ("fues_n_points_to_scan" in tuning_params)
        & (upper_envelope["method"] == "fues")
        else 10
    )

    upper_envelope["tuning_params"] = tuning_params
    processed_model_config["upper_envelope"] = upper_envelope

    if (
        continuous_states_info["n_additional_continuous_states"] > 1
        and upper_envelope["method"] != "druedahl_jorgensen"
    ):
        raise ValueError(
            "If more than one additional continuous state is specified, "
            "use upper_envelope['method'] = 'druedahl_jorgensen'."
        )

    if upper_envelope["method"] == "druedahl_jorgensen":
        if "assets_begin_of_period" not in model_config["continuous_states"]:
            raise ValueError(
                "Specify 'assets_begin_of_period' in model_config['continuous_states'] when using "
                "the 'druedahl_jorgensen' upper envelope method."
            )
        processed_model_config["continuous_states_info"]["assets_begin_of_period"] = (
            jnp.asarray(model_config["continuous_states"]["assets_begin_of_period"])
        )

    if upper_envelope["method"] == "fues":
        processed_model_config["n_total_wealth_grid"] = tuning_params[
            "n_total_wealth_grid"
        ]
    elif upper_envelope["method"] == "druedahl_jorgensen":
        # Expected value at 0, so add 1
        processed_model_config["n_total_wealth_grid"] = (
            len(model_config["continuous_states"]["assets_begin_of_period"]) + 1
        )
    else:
        raise ValueError("Something wrong internally")

    if "min_period_batch_segments" in model_config.keys():
        processed_model_config["min_period_batch_segments"] = model_config[
            "min_period_batch_segments"
        ]
    else:
        processed_model_config["min_period_batch_segments"] = None

    if "batch_mode" in model_config.keys():
        batch_mode = model_config["batch_mode"]
        valid_batch_modes = {"largest_block", "period_max"}
        if not isinstance(batch_mode, (str, list)):
            raise ValueError("batch_mode must be a string or a list of strings.")

        if isinstance(batch_mode, str):
            if batch_mode not in valid_batch_modes:
                raise ValueError(
                    f"batch_mode must be one of {valid_batch_modes}. Got {batch_mode}."
                )
        else:
            if not all(isinstance(mode, str) for mode in batch_mode):
                raise ValueError(
                    "If batch_mode is a list, all entries must be strings."
                )
            if not all(mode in valid_batch_modes for mode in batch_mode):
                raise ValueError(
                    f"All entries in batch_mode must be one of {valid_batch_modes}."
                )

            min_period_batch_segments = processed_model_config[
                "min_period_batch_segments"
            ]
            if min_period_batch_segments is None:
                expected_n_segments = 1
            elif isinstance(min_period_batch_segments, int):
                expected_n_segments = 2
            elif isinstance(min_period_batch_segments, list):
                expected_n_segments = len(min_period_batch_segments) + 1
            else:
                raise ValueError(
                    "min_period_batch_segments must be None, int, or list."
                )

            if len(batch_mode) != expected_n_segments:
                raise ValueError(
                    "If batch_mode is a list, it must have one entry per segment. "
                    f"Expected {expected_n_segments}, got {len(batch_mode)}."
                )

        processed_model_config["batch_mode"] = batch_mode
    else:
        processed_model_config["batch_mode"] = "largest_block"

    if "stochastic_states" in model_config.keys():
        processed_model_config["stochastic_states"] = model_config["stochastic_states"]

    if "deterministic_states" in model_config.keys():
        processed_model_config["deterministic_states"] = model_config[
            "deterministic_states"
        ]

    processed_model_config["params_check_info"] = {}

    return processed_model_config
