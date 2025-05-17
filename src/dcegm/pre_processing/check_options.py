import numpy as np


def check_model_config_and_process(model_config):
    """Check if options are valid and set defaults."""

    processed_model_config = {}

    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary.")

    if "n_periods" not in model_config:
        raise ValueError("model_config must contain the n_periods.")

    if not isinstance(model_config["n_periods"], int):
        raise ValueError("Number of periods must be an integer.")

    if not model_config["n_periods"] > 1:
        raise ValueError("Number of periods must be greater than 1.")

    processed_model_config["n_periods"] = model_config["n_periods"]

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
        raise ValueError("Choices must be given in model_config")

    if "continuous_states" not in model_config:
        raise ValueError("model_config must contain continuous_states as key.")

    continuous_state_grids = model_config["continuous_states"].copy()

    if not isinstance(continuous_state_grids, dict):
        raise ValueError("model_config['continuous_states'] must be a dictionary.")

    if "wealth" not in continuous_state_grids:
        raise ValueError(
            "model_config['continuous_states'] must contain wealth as key."
        )
    # Check if it is an array
    if not isinstance(continuous_state_grids["wealth"], (list, np.ndarray)):
        raise ValueError(
            "model_config['continuous_states']['wealth'] must be a list or an array."
        )

    # ToDo: Check if it is monotonic increasing

    continuous_states_info = {}
    n_savings_grid_points = len(continuous_state_grids)
    continuous_states_info["n_wealth_grid"] = n_savings_grid_points

    if len(continuous_state_grids) > 2:
        raise ValueError("At most two continuous states are supported.")

    elif len(continuous_state_grids) == 2:
        second_continuous_state = next(
            (
                {key: value}
                for key, value in model_config["continuous_states"].items()
                if key != "wealth"
            ),
            None,
        )

        continuous_states_info["second_continuous_exists"] = True

        second_continuous_state_name = list(second_continuous_state.keys())[0]
        continuous_states_info["second_continuous_state_name"] = (
            second_continuous_state_name
        )

        second_continuous_state_grid = continuous_state_grids[
            second_continuous_state_name
        ]
        # ToDo: Check if grid is array or list and monotonic increasing

        continuous_states_info["n_second_continuous_grid"] = len(
            second_continuous_state_grid
        )

    else:
        continuous_states_info["second_continuous_exists"] = False
        continuous_states_info["second_continuous_state_name"] = None
        continuous_states_info["n_second_continuous_grid"] = None

    processed_model_config["continuous_states_info"] = continuous_states_info

    if "exogenous_states" not in model_config:

    if "tuning_params" not in model_config:
        tuning_params = {}
    else:
        tuning_params = model_config["tuning_params"]

    tuning_params["extra_wealth_grid_factor"] = (
        tuning_params["extra_wealth_grid_factor"]
        if "extra_wealth_grid_factor" in tunin = Noneg_params
        else 0.2
    )
    tuning_params["n_constrained_points_to_add"] = (
        tuning_params["n_constrained_points_to_add"]
        if "n_constrained_points_to_add" in tuning_params
        else n_savings_grid_points // 10
    )

    if (
        n_savings_grid_points * (1 + tuning_params["extra_wealth_grid_factor"])
        < n_savings_grid_points + tuning_params["n_constrained_points_to_add"]
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
        n_savings_grid_points * (1 + tuning_params["extra_wealth_grid_factor"])
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

    processed_model_config["tuning_params"] = (tuning_params,)
    return processed_model_config
