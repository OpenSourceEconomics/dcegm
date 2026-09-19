import copy

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

    # How many income-shock draws the backward induction interpolates at once (see
    # solve_single_period.py). Not given means all of them in one block, which is the
    # single-pass solve; a smaller block size lowers the peak memory of that step.
    n_quad_points = model_config["n_quad_points"]
    income_shock_batch_size = model_config.get("income_shock_batch_size", None)
    if income_shock_batch_size is None:
        income_shock_batch_size = n_quad_points
    elif (
        not isinstance(income_shock_batch_size, int)
        or isinstance(income_shock_batch_size, bool)
        or not 1 <= income_shock_batch_size <= n_quad_points
        or n_quad_points % income_shock_batch_size != 0
    ):
        raise ValueError(
            "income_shock_batch_size must be None or an integer that divides "
            f"n_quad_points ({n_quad_points}), got {income_shock_batch_size!r}."
        )
    processed_model_config["income_shock_batch_size"] = income_shock_batch_size
    processed_model_config["n_income_shock_blocks"] = (
        n_quad_points // income_shock_batch_size
    )

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
    # A name left as `None` here has no default grid at all -- it must be paired
    # with a continuous_grid_functions entry (validated in
    # process_continuous_grid_functions, which is the first place both are known
    # together) and is fully state-choice-specific, with no global fallback array.
    continuous_states_info["additional_continuous_state_grids"] = {
        key: (None if value is None else jnp.asarray(value))
        for key, value in additional_continuous_states.items()
    }
    continuous_states_info["n_additional_continuous_states"] = len(
        additional_continuous_states
    )
    continuous_states_info["has_additional_continuous_state"] = (
        continuous_states_info["n_additional_continuous_states"] > 0
    )
    # Names declared as `None` -- their size can only be pinned once a real
    # state-choice exists to evaluate the grid function against (see
    # continuous_state_grids.py's evaluate_state_specific_continuous_grids, run once
    # the state-choice space is built), so n_continuous_state_combinations is left
    # unresolved (None) here whenever any of these are present; resolved later and
    # merged back into this dict.
    continuous_states_info["state_specific_size_names"] = [
        key
        for key, value in continuous_states_info[
            "additional_continuous_state_grids"
        ].items()
        if value is None
    ]
    if continuous_states_info["state_specific_size_names"]:
        # Number of combo points spanned by the additional continuous states' grids
        # can't be known yet -- at least one dimension's length is pending.
        continuous_states_info["n_continuous_state_combinations"] = None
    else:
        # Number of combo points spanned by the additional continuous states' default
        # grids (1 -- the dummy placeholder -- when there are none). Only the *count*
        # is needed here, for sizing solution containers: every state-choice's own
        # grid for a given name has this same length by construction (state-specific
        # grids may vary in value, not size), regardless of which values it holds.
        continuous_states_info["n_continuous_state_combinations"] = int(
            np.prod(
                [
                    len(grid)
                    for grid in continuous_states_info[
                        "additional_continuous_state_grids"
                    ].values()
                ]
            )
        )

    processed_model_config["continuous_states_info"] = continuous_states_info

    # Set default upper envelope method if not given.
    if "upper_envelope" not in model_config:
        upper_envelope = {}
        upper_envelope["method"] = "fues"
    elif "method" not in model_config["upper_envelope"]:
        upper_envelope = copy.deepcopy(model_config["upper_envelope"])
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
        upper_envelope = copy.deepcopy(model_config["upper_envelope"])

    if "tuning_params" not in upper_envelope:
        tuning_params = {}
    elif "tuning_params" in model_config:
        raise ValueError(
            "tuning_params should be nested in model_config['upper_envelope']"
        )
    else:
        tuning_params = upper_envelope["tuning_params"]

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
        assets_begin_of_period_grid = model_config["continuous_states"][
            "assets_begin_of_period"
        ]
        # `None` when the grid is fully state-choice-specific (paired with a
        # continuous_grid_functions entry, validated in
        # process_continuous_grid_functions); its length is then pinned later, once
        # a real state-choice can be evaluated against. Otherwise store the declared
        # array, used to pin the wealth-grid length (see continuous_state_grids.py).
        # Either way, the actual per-state-choice wealth grid the Druedahl-Jorgensen
        # upper envelope evaluates on -- and that every reader interpolates against
        # -- is recomputed on demand from
        # continuous_grid_functions["assets_begin_of_period"] (see
        # compute_own_dj_wealth_grid), so no shared grid is stored here.
        processed_model_config["continuous_states_info"]["assets_begin_of_period"] = (
            None
            if assets_begin_of_period_grid is None
            else jnp.asarray(assets_begin_of_period_grid)
        )

    if upper_envelope["method"] == "fues":
        if "assets_begin_of_period" in model_config["continuous_states"]:
            raise ValueError(
                "'assets_begin_of_period' is only used by the 'druedahl_jorgensen' "
                "upper envelope method. It was found in "
                "model_config['continuous_states'] together with "
                "upper_envelope['method'] == 'fues', where it has no effect -- "
                "either remove it or switch to "
                "upper_envelope['method'] = 'druedahl_jorgensen'."
            )
        processed_model_config["n_total_wealth_grid"] = tuning_params[
            "n_total_wealth_grid"
        ]
    elif upper_envelope["method"] == "druedahl_jorgensen":
        # Expected value at 0, so add 1. None when assets_begin_of_period is
        # state-specific (declared as `None`) -- resolved later, once its size can
        # be pinned by evaluating the grid function against a real state-choice.
        assets_begin_of_period_grid = model_config["continuous_states"][
            "assets_begin_of_period"
        ]
        processed_model_config["n_total_wealth_grid"] = (
            None
            if assets_begin_of_period_grid is None
            else len(assets_begin_of_period_grid) + 1
        )
    else:
        raise ValueError("Something wrong internally")

    # With a single discrete choice, the upper envelope is skipped entirely (see
    # create_upper_envelope_function), so the stored endog_grid is not the fixed
    # Druedahl-Jorgensen grid in that case and must still be stored/read normally.
    upper_envelope["skip_endog_grid_storage"] = (
        upper_envelope["method"] == "druedahl_jorgensen"
        and len(processed_model_config["choices"]) >= 2
    )

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
