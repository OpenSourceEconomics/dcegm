"""Assemble one big model's solution from several smaller sub-model solves.

When the pooled solve of a model does not fit into (GPU) memory, the model can be split
by hand into several smaller ``setup_model`` instances whose discrete state-choice
spaces together tile the big model's state-choice space exactly -- e.g. one sub-model
per invariant "type" (education x sex): because such types never change along any
transition or proxy, the problem is block-diagonal in them and each block is an exact,
standalone sub-problem.

``get_solve_from_small_models`` builds the big model, checks that the supplied sub-
models really are such a tiling, and returns a solve function that solves each sub-model
and scatters its solution into the big solution container. Sequentially, only one
block's device arrays are ever live, so the big solution never has to be solved (only
stored) in one piece.

"""

import warnings
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from dcegm.interfaces.index_functions import (
    get_state_choice_index_per_discrete_states_and_choices,
)
from dcegm.interfaces.model_class import setup_model
from dcegm.interfaces.sol_interface import model_solved


def get_solve_from_small_models(
    small_models: List[setup_model],
    parallel: bool,
    model_config: Optional[Dict] = None,
    model_specs: Optional[Dict] = None,
    utility_functions: Optional[Dict[str, Callable]] = None,
    utility_functions_final_period: Optional[Dict[str, Callable]] = None,
    budget_constraint: Optional[Callable] = None,
    state_space_functions: Optional[Dict[str, Callable]] = None,
    stochastic_states_transitions: Optional[Dict[str, Callable]] = None,
    shock_functions: Optional[Dict[str, Callable]] = None,
    continuous_grid_functions: Optional[Dict[str, Callable]] = None,
    alternative_sim_specifications: Optional[Dict[str, Callable]] = None,
    debug_info: Optional[str] = None,
    use_stochastic_sparsity: bool = False,
    big_model: Optional[setup_model] = None,
) -> Callable[[Dict[str, jnp.ndarray]], model_solved]:
    """Return a solve function that assembles the big solve from sub-model solves.

    The big model is either supplied via ``big_model`` (already built) or built here
    from ``model_config`` plus the remaining ``setup_model`` build arguments.

    Args:
        small_models: The sub-models to solve. Their discrete state-choice spaces must
            tile the big model's state-choice space exactly (partition -- every big
            state-choice covered by exactly one sub-model). Build them with the same
            model functions as the big model, restricting only the invariant type
            state(s) (see the module docstring).
        parallel: If False, solve the sub-models one at a time in a Python loop, so
            only one block's device arrays are live at once (the memory-saving path).
            If True, dispatch the sub-model solves across ``jax.devices()`` (one block
            per device, round-robin) and read them back together; this holds several
            blocks' arrays at once and only speeds things up with more than one device.
        big_model: An already-built big (full) model. If given, the remaining build
            arguments are ignored. If omitted, the big model is built from
            ``model_config`` and the build arguments below.
        model_config: Config of the big (full) model (required when ``big_model`` is
            omitted).
        model_specs, utility_functions, utility_functions_final_period,
        budget_constraint, state_space_functions, stochastic_states_transitions,
        shock_functions, continuous_grid_functions, alternative_sim_specifications,
        debug_info, use_stochastic_sparsity: The remaining ``setup_model`` build
            arguments for the big model; forward the same objects used for the
            sub-models.

    Returns:
        A function ``solve(params) -> model_solved`` for the big model, whose
        ``value``/``policy``/``endog_grid`` are the sub-model solves scattered into the
        big solution container. The arrays are host (NumPy) arrays: the full solution
        is stored on the host and never materialised on device in one piece.

    """
    if big_model is None:
        if model_config is None:
            raise ValueError(
                "Provide either a pre-built big_model or model_config (plus the "
                "remaining setup_model build arguments) to build the big model."
            )
        big_model = setup_model(
            model_config=model_config,
            model_specs=model_specs,
            utility_functions=utility_functions,
            utility_functions_final_period=utility_functions_final_period,
            budget_constraint=budget_constraint,
            state_space_functions=state_space_functions,
            stochastic_states_transitions=stochastic_states_transitions,
            shock_functions=shock_functions,
            continuous_grid_functions=continuous_grid_functions,
            alternative_sim_specifications=alternative_sim_specifications,
            debug_info=debug_info,
            use_stochastic_sparsity=use_stochastic_sparsity,
        )

    _check_small_models_consistency(big_model, small_models)
    big_rows_per_model = _compute_big_rows_per_model(big_model, small_models)
    _check_partition(big_model, big_rows_per_model)

    # Compile each sub-model's solve once; reused across params.
    small_solve_funcs = [m.get_solve_func() for m in small_models]

    store_endog = not big_model.model_config["upper_envelope"][
        "skip_endog_grid_storage"
    ]
    n_state_choices = big_model.model_structure["state_choice_space"].shape[0]
    n_continuous = big_model.model_config["continuous_states_info"][
        "n_continuous_state_combinations"
    ]
    n_wealth = big_model.model_config["n_total_wealth_grid"]
    container_shape = (n_state_choices, n_continuous, n_wealth)

    def solve_from_small(params: Dict[str, jnp.ndarray]) -> model_solved:
        value = np.full(container_shape, np.nan)
        policy = np.full(container_shape, np.nan)
        endog_grid = np.full(container_shape, np.nan) if store_endog else None

        if parallel:
            solutions = _dispatch_parallel(small_solve_funcs, params)
        else:
            solutions = None

        for i, rows in enumerate(big_rows_per_model):
            sol = solutions[i] if parallel else small_solve_funcs[i](params)
            # np.asarray pulls the block off device and scatters it into the host
            # container; the block's device arrays are then free to be released.
            value[rows] = np.asarray(sol.value)
            policy[rows] = np.asarray(sol.policy)
            if store_endog:
                endog_grid[rows] = np.asarray(sol.endog_grid)

        return model_solved(
            model=big_model,
            params=params,
            value=value,
            policy=policy,
            endog_grid=endog_grid,
        )

    return solve_from_small


def _dispatch_parallel(small_solve_funcs, params):
    """Dispatch each sub-model solve on its own device (round-robin) without blocking.

    Returns the (still async) ``model_solved`` results; the caller blocks on them when
    it reads the arrays out. With a single device this is effectively sequential.

    """
    devices = jax.devices()
    if len(small_solve_funcs) > len(devices):
        warnings.warn(
            f"parallel=True dispatches {len(small_solve_funcs)} sub-model solves "
            f"across {len(devices)} device(s); with fewer devices than sub-models the "
            "solves cannot all run concurrently and several blocks are held in memory "
            "at once. Use parallel=False to keep only one block live at a time.",
            stacklevel=2,
        )

    results = []
    for i, solve_func in enumerate(small_solve_funcs):
        with jax.default_device(devices[i % len(devices)]):
            results.append(solve_func(params))
    return results


def _compute_big_rows_per_model(big_model, small_models):
    """For each sub-model, the big state-choice row of each of its state-choices."""
    big_structure = big_model.model_structure
    discrete_states_names = big_structure["discrete_states_names"]

    big_rows_per_model = []
    for model in small_models:
        state_choice_space_dict = model.model_structure["state_choice_space_dict"]
        states = {
            name: jnp.asarray(state_choice_space_dict[name])
            for name in discrete_states_names
        }
        choices = jnp.asarray(state_choice_space_dict["choice"])
        big_rows = get_state_choice_index_per_discrete_states_and_choices(
            model_structure=big_structure,
            states=states,
            choices=choices,
        )
        big_rows_per_model.append(np.asarray(big_rows))
    return big_rows_per_model


def _check_partition(big_model, big_rows_per_model):
    """Check the sub-models tile the big state-choice space exactly."""
    big_structure = big_model.model_structure
    n_state_choices = big_structure["state_choice_space"].shape[0]

    indexer_dtype = np.asarray(
        big_structure["map_state_choice_to_index_with_proxy"]
    ).dtype
    invalid_index = np.iinfo(indexer_dtype).max

    all_rows = np.concatenate([np.asarray(rows) for rows in big_rows_per_model])

    if np.any(all_rows == invalid_index):
        raise ValueError(
            "Some sub-model state-choices do not exist in the big model (their "
            "state-choice lookup returned the invalid index). The sub-models must be "
            "restrictions of the big model built with the same state-space functions."
        )

    if all_rows.shape[0] != n_state_choices:
        raise ValueError(
            "The sub-models' state-choices do not sum to the big model's "
            f"state-choices: got {all_rows.shape[0]} across sub-models, but the big "
            f"model has {n_state_choices} state-choices."
        )

    if not np.array_equal(np.sort(all_rows), np.arange(n_state_choices)):
        raise ValueError(
            "The sub-models do not partition the big model's state-choice space "
            "exactly: their mapped rows are not a permutation of all big rows (some "
            "big state-choices are covered by more than one sub-model or by none)."
        )


def _check_small_models_consistency(big_model, small_models):
    """Check the sub-models are shape- and structure-compatible with the big model."""
    if len(small_models) == 0:
        raise ValueError("small_models must be a non-empty list of models.")

    big_config = big_model.model_config
    big_structure = big_model.model_structure
    big_continuous = big_config["continuous_states_info"]

    for i, model in enumerate(small_models):
        config = model.model_config
        structure = model.model_structure
        continuous = config["continuous_states_info"]
        prefix = f"Sub-model {i}:"

        if structure["discrete_states_names"] != big_structure["discrete_states_names"]:
            raise ValueError(
                f"{prefix} discrete_states_names differ from the big model: "
                f"{structure['discrete_states_names']} vs "
                f"{big_structure['discrete_states_names']}."
            )

        if not np.array_equal(
            np.asarray(structure["choice_range"]),
            np.asarray(big_structure["choice_range"]),
        ):
            raise ValueError(f"{prefix} choices differ from the big model.")

        if config["n_periods"] != big_config["n_periods"]:
            raise ValueError(f"{prefix} n_periods differs from the big model.")

        if config["n_quad_points"] != big_config["n_quad_points"]:
            raise ValueError(f"{prefix} n_quad_points differs from the big model.")

        if config["n_total_wealth_grid"] != big_config["n_total_wealth_grid"]:
            raise ValueError(
                f"{prefix} n_total_wealth_grid differs from the big model "
                f"({config['n_total_wealth_grid']} vs "
                f"{big_config['n_total_wealth_grid']})."
            )

        if (
            continuous["n_continuous_state_combinations"]
            != big_continuous["n_continuous_state_combinations"]
        ):
            raise ValueError(
                f"{prefix} n_continuous_state_combinations differs from the big model."
            )

        if not np.allclose(
            np.asarray(continuous["assets_grid_end_of_period"]),
            np.asarray(big_continuous["assets_grid_end_of_period"]),
        ):
            raise ValueError(
                f"{prefix} assets_end_of_period grid differs from the big model."
            )

        if (
            continuous["additional_continuous_state_names"]
            != big_continuous["additional_continuous_state_names"]
        ):
            raise ValueError(
                f"{prefix} additional continuous state names differ from the big model."
            )
        for name, grid in big_continuous["additional_continuous_state_grids"].items():
            small_grid = continuous["additional_continuous_state_grids"][name]
            if (grid is None) != (small_grid is None):
                raise ValueError(
                    f"{prefix} additional continuous grid '{name}' differs from the "
                    "big model (one is state-specific and the other is not)."
                )
            if grid is not None and not np.allclose(
                np.asarray(small_grid), np.asarray(grid)
            ):
                raise ValueError(
                    f"{prefix} additional continuous grid '{name}' differs from the "
                    "big model."
                )

        if config["upper_envelope"]["method"] != big_config["upper_envelope"]["method"]:
            raise ValueError(
                f"{prefix} upper_envelope method differs from the big model."
            )

        if (
            config["upper_envelope"]["skip_endog_grid_storage"]
            != big_config["upper_envelope"]["skip_endog_grid_storage"]
        ):
            raise ValueError(
                f"{prefix} skip_endog_grid_storage differs from the big model."
            )

        if (
            structure["stochastic_states_names"]
            != big_structure["stochastic_states_names"]
        ):
            raise ValueError(
                f"{prefix} stochastic_states_names differ from the big model."
            )

        if not np.array_equal(
            np.asarray(structure["stochastic_state_space"]),
            np.asarray(big_structure["stochastic_state_space"]),
        ):
            raise ValueError(
                f"{prefix} stochastic_state_space differs from the big model."
            )
