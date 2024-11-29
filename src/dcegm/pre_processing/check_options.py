import numpy as np


def check_options_and_set_defaults(options):
    """Check if options are valid and set defaults."""

    if not isinstance(options, dict):
        raise ValueError("Options must be a dictionary.")

    if "state_space" in options:
        if not isinstance(options["state_space"], dict):
            raise ValueError("State space must be a dictionary.")
    else:
        raise ValueError("Options must contain a state space dictionary.")

    if "n_periods" not in options["state_space"]:
        raise ValueError("State space must contain the number of periods.")

    if not isinstance(options["state_space"]["n_periods"], int):
        raise ValueError("Number of periods must be an integer.")

    if "choices" not in options["state_space"]:
        print("Choices not given. Assume only single choice with value 0")
        options["state_space"]["choices"] = np.array([0], dtype=np.uint8)

    if "choices" in options["state_space"]:
        if isinstance(options["state_space"]["choices"], list):
            options["state_space"]["choices"] = np.array(
                options["state_space"]["choices"], dtype=np.uint8
            )
        elif isinstance(options["state_space"]["choices"], int):
            options["state_space"]["choices"] = np.array(
                [options["state_space"]["choices"]], dtype=np.uint8
            )
        elif isinstance(options["state_space"]["choices"], np.ndarray):
            options["state_space"]["choices"] = options["state_space"][
                "choices"
            ].astype(np.uint8)
        else:
            raise ValueError("Choices must be a list or an integer.")

    if "model_params" not in options:
        raise ValueError("Options must contain a model parameters dictionary.")

    if not isinstance(options["model_params"], dict):
        raise ValueError("Model parameters must be a dictionary.")

    if "n_choices" not in options["model_params"]:
        options["model_params"]["n_choices"] = len(options["state_space"]["choices"])

    n_savings_grid_points = len(options["state_space"]["continuous_states"]["wealth"])
    options["n_wealth_grid"] = n_savings_grid_points

    if "tuning_params" not in options:
        options["tuning_params"] = {}

    options["tuning_params"]["extra_wealth_grid_factor"] = (
        options["tuning_params"]["extra_wealth_grid_factor"]
        if "extra_wealth_grid_factor" in options["tuning_params"]
        else 0.2
    )
    options["tuning_params"]["n_constrained_points_to_add"] = (
        options["tuning_params"]["n_constrained_points_to_add"]
        if "n_constrained_points_to_add" in options["tuning_params"]
        else n_savings_grid_points // 10
    )

    if (
        n_savings_grid_points
        * (1 + options["tuning_params"]["extra_wealth_grid_factor"])
        < n_savings_grid_points
        + options["tuning_params"]["n_constrained_points_to_add"]
    ):
        raise ValueError(
            f"""\n\n
            When preparing the tuning parameters for the upper
            envelope, we found the following contradicting parameters: \n
            The extra wealth grid factor of {options["tuning_params"]["extra_wealth_grid_factor"]} is too small
            to cover the {options["tuning_params"]["n_constrained_points_to_add"]} wealth points which are added in
            the credit constrained part of the wealth grid. \n\n"""
        )
    options["tuning_params"]["n_total_wealth_grid"] = int(
        n_savings_grid_points
        * (1 + options["tuning_params"]["extra_wealth_grid_factor"])
    )

    exog_grids = options["state_space"]["continuous_states"].copy()

    if len(options["state_space"]["continuous_states"]) == 2:
        second_continuous_state = next(
            (
                {key: value}
                for key, value in options["state_space"]["continuous_states"].items()
                if key != "wealth"
            ),
            None,
        )

        second_continuous_state_name = list(second_continuous_state.keys())[0]
        options["second_continuous_state_name"] = second_continuous_state_name

        options["tuning_params"]["n_second_continuous_grid"] = len(
            second_continuous_state[second_continuous_state_name]
        )

        exog_grids["second_continuous"] = options["state_space"]["continuous_states"][
            second_continuous_state_name
        ]
        exog_grids.pop(second_continuous_state_name)

    options["exog_grids"] = exog_grids

    return options
