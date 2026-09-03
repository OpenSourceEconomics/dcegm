from typing import Any, Callable, Dict, Tuple

from jax import numpy as jnp
from jax import vmap

from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp1d_dj import interp1d_policy_and_value_on_wealth_dj
from dcegm.interpolation.interp2d_irregular import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from dcegm.interpolation.interpnd_regular import (
    interpnd_policy_and_value_for_child_states_on_own_regular_grids,
)
from dcegm.law_of_motion import (
    calc_law_of_motion_for_state_choices,
    compute_own_continuous_grid_combos,
    compute_own_continuous_grids_raw,
    compute_own_dj_wealth_grid,
)


def interpolate_value_and_marg_util(
    model_funcs,
    state_choice_vec: Dict[str, int],
    continuous_grids_info: Dict[str, Any],
    income_shocks_scaled: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
    upper_envelope_method: str,
    skip_endog_grid_storage: bool,
    grid_source_state_choice_vec: Dict[str, int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy for all child states and compute marginal utility.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility of consumption.
        compute_utility (callable): Function for calculating the utility of consumption.
        state_choice_vec (dict): Dictionary containing the state and choice of the agent
            (the child, in the main solve path).
        grid_source_state_choice_vec (dict): State-choice dict of a representative
            parent, used only to select which state's own continuous grid to feed
            the law of motion -- see the docstring of
            calc_law_of_motion_for_state_choices for why this must be the parent,
            not state_choice_vec (the child). Falls back to state_choice_vec itself
            when not given (today's global-grid behavior).
        assets_beginning_of_next_period (jnp.ndarray): 2d array of shape
            (n_quad_stochastic, n_grid_wealth,) containing the agent's beginning of
            period wealth.
        endog_grid_child_state_choice (jnp.ndarray): 1d array containing the endogenous
            wealth grid of the child state/choice pair. Shape (n_grid_wealth,).
        policy_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding policy function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        value_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding value function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        has_second_continuous_state (bool): Boolean indicating whether the model
            features a second continuous state variable. If False, the only
            continuous state variable is consumption/savings.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - value_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated value function.
        - marg_util_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated marginal utilities for each wealth level and
            income shock.

    """
    # Check if interpolation needs to be multidimensional and irregular
    multi_dim = continuous_grids_info["has_additional_continuous_state"]
    irregular = upper_envelope_method == "fues"

    # Compute the child continuous-state/wealth transitions on demand for exactly
    # this batch's children, instead of reading from a precomputed whole-state-space
    # structure (see law_of_motion.py).
    law_of_motion = calc_law_of_motion_for_state_choices(
        state_choice_vec=state_choice_vec,
        income_shocks_scaled=income_shocks_scaled,
        params=params,
        model_funcs=model_funcs,
        has_additional_continuous_states=multi_dim,
        additional_continuous_state_names=continuous_grids_info[
            "additional_continuous_state_names"
        ],
        grid_source_state_choice_vec=grid_source_state_choice_vec,
    )
    wealth_child_states = law_of_motion["assets_begin_of_period"]
    continuous_states_next = law_of_motion["continuous_states"]

    compute_marginal_utility = model_funcs["compute_marginal_utility"]
    compute_utility = model_funcs["compute_utility"]
    discount_factor = model_funcs["read_funcs"]["discount_factor"](params)

    if multi_dim & irregular:
        return _interpolate_value_and_marg_util_2d_irregular(
            compute_marginal_utility=compute_marginal_utility,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            continuous_grids_info=continuous_grids_info,
            continuous_states_next=continuous_states_next,
            wealth_child_states=wealth_child_states,
            endog_grid_child_state_choice=endog_grid_child_state_choice,
            policy_child_state_choice=policy_child_state_choice,
            value_child_state_choice=value_child_state_choice,
            params=params,
            discount_factor=discount_factor,
            continuous_grid_functions=model_funcs["continuous_grid_functions"],
        )

    elif multi_dim & (not irregular):
        return _interpolate_value_and_marg_util_nd_regular(
            compute_marginal_utility=compute_marginal_utility,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            continuous_grids_info=continuous_grids_info,
            continuous_states_next=continuous_states_next,
            wealth_child_states=wealth_child_states,
            endog_grid_child_state_choice=endog_grid_child_state_choice,
            policy_child_state_choice=policy_child_state_choice,
            value_child_state_choice=value_child_state_choice,
            params=params,
            discount_factor=discount_factor,
            skip_endog_grid_storage=skip_endog_grid_storage,
            continuous_grid_functions=model_funcs["continuous_grid_functions"],
        )
    else:
        # Selects inside if jorgensen_druedahl or fues (different treatment of budget constraint)
        # Under DJ, each child's own wealth grid, evaluated on demand -- state_choice_vec
        # is the CHILD's own identity here (reading each child's own stored solution,
        # interpolated on its own grid), not a parent, same reasoning as
        # compute_own_continuous_grid_combos elsewhere in this file.
        if skip_endog_grid_storage:
            own_dj_wealth_grid = vmap(compute_own_dj_wealth_grid, in_axes=(0, None))(
                state_choice_vec, model_funcs["continuous_grid_functions"]
            )
            endog_grid_arg = jnp.broadcast_to(
                own_dj_wealth_grid[:, None, :],
                (
                    own_dj_wealth_grid.shape[0],
                    policy_child_state_choice.shape[1],
                    own_dj_wealth_grid.shape[1],
                ),
            )
            endog_grid_in_axes = 0
        else:
            endog_grid_arg = endog_grid_child_state_choice
            endog_grid_in_axes = 0

        interp_for_single_state_choice = vmap(
            interp1d_value_and_marg_util_for_state_choice,
            in_axes=(None, None, 0, 0, endog_grid_in_axes, 0, 0, None, None, None),
        )

        return interp_for_single_state_choice(
            compute_marginal_utility,
            compute_utility,
            state_choice_vec,
            wealth_child_states,
            endog_grid_arg,
            policy_child_state_choice,
            value_child_state_choice,
            params,
            discount_factor,
            upper_envelope_method == "druedahl_jorgensen",
        )


def interp1d_value_and_marg_util_for_state_choice(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    assets_beginning_of_next_period: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
    discount_factor: float,
    use_dj_interpolation: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy for given child state and compute marginal utility.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility of consumption.
        compute_utility (callable): Function for calculating the utility of consumption.
        state_choice_vec (dict): Dictionary containing the state and choice of the agent.
        assets_beginning_of_next_period (jnp.ndarray): 2d array of shape
            (n_quad_stochastic, n_grid_wealth,) containing the agent's beginning of
            period wealth.
        endog_grid_child_state_choice (jnp.ndarray): 1d array containing the endogenous
            wealth grid of the child state/choice pair. Shape (n_grid_wealth,).
        policy_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding policy function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        value_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding value function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        has_second_continuous_state (bool): Boolean indicating whether the model
            features a second continuous state variable. If False, the only
            continuous state variable is consumption/savings.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - marg_utils (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated marginal utilities for each wealth level and
            income shock.
        - value_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated value function.

    """
    endog_grid_child_state_choice = jnp.asarray(endog_grid_child_state_choice)
    policy_child_state_choice = jnp.asarray(policy_child_state_choice)
    value_child_state_choice = jnp.asarray(value_child_state_choice)
    assets_beginning_of_next_period = jnp.asarray(assets_beginning_of_next_period)

    def interp_on_single_wealth_point(wealth_point):
        if use_dj_interpolation:
            policy_interp, value_interp = interp1d_policy_and_value_on_wealth_dj(
                wealth=wealth_point,
                wealth_grid=endog_grid_child_state_choice[0],
                policy_grid=policy_child_state_choice[0],
                value_grid=value_child_state_choice[0],
                compute_utility=compute_utility,
                state_choice_vec=state_choice_vec,
                params=params,
                discount_factor=discount_factor,
            )
        else:
            policy_interp, value_interp = interp1d_policy_and_value_on_wealth(
                wealth=wealth_point,
                wealth_grid=endog_grid_child_state_choice[0],
                policy_grid=policy_child_state_choice[0],
                value_grid=value_child_state_choice[0],
                compute_utility=compute_utility,
                state_choice_vec=state_choice_vec,
                params=params,
                discount_factor=discount_factor,
            )
        marg_util_interp = compute_marginal_utility(
            consumption=policy_interp, params=params, **state_choice_vec
        )

        return value_interp, marg_util_interp

    interp_over_single_wealth_and_income_shock_draw = vmap(
        vmap(interp_on_single_wealth_point)  # income shocks
    )  # wealth grid

    # Select dummy dimension
    assets_points = jnp.asarray(assets_beginning_of_next_period)[0]
    value_interp, marg_util_interp = interp_over_single_wealth_and_income_shock_draw(
        assets_points
    )
    value_interp = jnp.asarray(value_interp)
    marg_util_interp = jnp.asarray(marg_util_interp)

    # Add it back in the beginning
    return value_interp[None, :, :], marg_util_interp[None, :, :]


def _interpolate_value_and_marg_util_2d_irregular(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    continuous_grids_info: Dict[str, Any],
    continuous_states_next: Dict[str, jnp.ndarray],
    wealth_child_states: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
    discount_factor: float,
    continuous_grid_functions: Dict[str, Callable],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and marginal utility on the irregular FUES 2D grid.

    Reached only for the FUES (irregular) upper envelope, which check_model_config.py
    restricts to at most one additional continuous state (n > 1 requires
    upper_envelope["method"] == "druedahl_jorgensen", handled by
    _interpolate_value_and_marg_util_nd_regular instead). So indexing [0] below covers
    the only additional continuous state that can exist on this path.

    """
    continuous_state_name = continuous_grids_info["additional_continuous_state_names"][
        0
    ]
    continuous_state_child_states = continuous_states_next[continuous_state_name]

    # Each child's own grid is now computed inside
    # interp2d_value_and_marg_util_for_state_choice, on demand, after vmapping
    # down to a single (child) state-choice -- see its docstring.
    interp_for_single_state_choice = vmap(
        interp2d_value_and_marg_util_for_state_choice,
        in_axes=(
            None,
            None,
            0,
            None,
            None,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
        ),
    )

    return interp_for_single_state_choice(
        compute_marginal_utility,
        compute_utility,
        state_choice_vec,
        continuous_grid_functions,
        continuous_grids_info["additional_continuous_state_names"],
        wealth_child_states,
        continuous_state_child_states,
        endog_grid_child_state_choice,
        policy_child_state_choice,
        value_child_state_choice,
        params,
        discount_factor,
    )


def _interpolate_value_and_marg_util_nd_regular(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    continuous_grids_info: Dict[str, Any],
    continuous_states_next: Dict[str, jnp.ndarray],
    wealth_child_states: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
    discount_factor: float,
    skip_endog_grid_storage: bool,
    continuous_grid_functions: Dict[str, Callable],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and marginal utility on the regular n-D grid.

    Reached for the Druedahl-Jorgensen upper envelope, which supports an arbitrary
    number of additional continuous states.

    """
    continuous_state_names = continuous_grids_info["additional_continuous_state_names"]
    continuous_state_child_states = {
        name: continuous_states_next[name] for name in continuous_state_names
    }

    # Each child's own grid on demand, instead of reading the shared/global
    # additional_continuous_state_grids -- see law_of_motion.py. state_choice_vec
    # is the CHILD's own identity here (we're reading each child's own stored
    # solution, interpolated on its own grid), not a parent -- unlike
    # law_of_motion.py's grid selection for the *transition into* a child, which
    # needs a representative parent. Unmeshed (per-dimension) grids, since
    # interpnd's index/weight math handles each dimension separately before
    # combining via strides/corner tables.
    own_continuous_state_grids_per_child = vmap(
        compute_own_continuous_grids_raw,
        in_axes=(0, None, None),
    )(
        state_choice_vec,
        continuous_grid_functions,
        continuous_state_names,
    )

    # Each child's own wealth grid on demand, same reasoning as
    # own_continuous_state_grids_per_child above: state_choice_vec is the CHILD's
    # own identity here, since we're reading each child's own stored solution
    # (mirrors the simple-1D-DJ branch's own_dj_wealth_grid computation). Only
    # reachable when skip_endog_grid_storage -- with a single choice the DJ upper
    # envelope is skipped entirely (see check_model_config.py), so the stored
    # endog_grid is a genuinely different (non-fixed) grid there instead, handled
    # by the else branch below (see process_continuous_grid_functions, which
    # forbids a state-specific assets_begin_of_period whenever
    # skip_endog_grid_storage is False and additional continuous states exist).
    wealth_grid = (
        vmap(compute_own_dj_wealth_grid, in_axes=(0, None))(
            state_choice_vec, continuous_grid_functions
        )
        if skip_endog_grid_storage
        else endog_grid_child_state_choice[0, 0]
    )

    policy_interp, value_interp = (
        interpnd_policy_and_value_for_child_states_on_own_regular_grids(
            additional_continuous_state_grids_per_child=own_continuous_state_grids_per_child,
            wealth_grid=wealth_grid,
            policy_grid_child_states=policy_child_state_choice,
            value_grid_child_states=value_child_state_choice,
            continuous_state_child_states=continuous_state_child_states,
            wealth_child_states=wealth_child_states,
            state_choice_child_states=state_choice_vec,
            compute_utility=compute_utility,
            params=params,
            discount_factor=discount_factor,
        )
    )

    marg_util_interp = _compute_nd_marginal_utility(
        compute_marginal_utility=compute_marginal_utility,
        policy_interp=policy_interp,
        state_choice_child_states=state_choice_vec,
        continuous_state_child_states=continuous_state_child_states,
        params=params,
    )
    return value_interp, marg_util_interp


def _compute_nd_marginal_utility(
    compute_marginal_utility: Callable,
    policy_interp: jnp.ndarray,
    state_choice_child_states: Dict[str, Any],
    continuous_state_child_states: Dict[str, jnp.ndarray],
    params: Dict[str, float],
) -> jnp.ndarray:
    """Compute marginal utility pointwise on ND interpolation output via vmaps."""
    state_choice_child_states_marg = {
        key: jnp.asarray(value) for key, value in state_choice_child_states.items()
    }
    continuous_state_child_states_marg = {
        key: jnp.asarray(value) for key, value in continuous_state_child_states.items()
    }

    def _marg_util_at_point(
        consumption_point,
        state_choice_point,
        continuous_state_point,
    ):
        out = compute_marginal_utility(
            consumption=consumption_point,
            params=params,
            **state_choice_point,
            **continuous_state_point,
        )
        if isinstance(out, tuple):
            out = out[0]
        return jnp.asarray(out)

    return vmap(
        vmap(
            vmap(
                vmap(
                    _marg_util_at_point,
                    in_axes=(0, None, None),
                ),
                in_axes=(0, None, None),
            ),
            in_axes=(0, None, 0),
        ),
        in_axes=(0, 0, 0),
    )(
        policy_interp,
        state_choice_child_states_marg,
        continuous_state_child_states_marg,
    )


def interp2d_value_and_marg_util_for_state_choice(
    compute_marginal_utility: Callable,
    compute_utility: Callable,
    state_choice_vec: Dict[str, int],
    continuous_grid_functions: Dict[str, Callable],
    additional_continuous_state_names,
    assets_beginning_of_next_period: jnp.ndarray,
    continuous_state_beginning_of_next_period: jnp.ndarray,
    endog_grid_child_state_choice: jnp.ndarray,
    policy_child_state_choice: jnp.ndarray,
    value_child_state_choice: jnp.ndarray,
    params: Dict[str, float],
    discount_factor: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate value and policy for given child state and compute marginal utility.

    Args:
        compute_marginal_utility (callable): User-defined function to compute the
            agent's marginal utility of consumption.
        compute_utility (callable): Function for calculating the utility of consumption.
        state_choice_vec (dict): Dictionary containing the state and choice of the agent.
        assets_beginning_of_next_period (jnp.ndarray): 2d array of shape
            (n_quad_stochastic, n_grid_wealth,) containing the agent's beginning of
            period wealth.
        endog_grid_child_state_choice (jnp.ndarray): 1d array containing the endogenous
            wealth grid of the child state/choice pair. Shape (n_grid_wealth,).
        policy_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding policy function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        value_child_state_choice (jnp.ndarray): 1d array containing the
            corresponding value function values of the endogenous wealth grid of the
            child state/choice pair. Shape (n_grid_wealth,).
        has_second_continuous_state (bool): Boolean indicating whether the model
            features a second continuous state variable. If False, the only
            continuous state variable is consumption/savings.
        params (dict): Dictionary containing the model parameters.

    Returns:
        tuple:

        - marg_utils (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated marginal utilities for each wealth level and
            income shock.
        - value_interp (jnp.ndarray): 2d array of shape (n_wealth_grid, n_income_shocks)
            containing the interpolated value function.

    """
    # This state-choice's own grid, on demand -- computed here, after vmapping
    # down to a single (child) state-choice, instead of precomputing the whole
    # batch's grids upfront in a separate vmap and feeding the result in as a
    # paired array. state_choice_vec is the CHILD's own identity here (we're
    # reading this child's own stored solution, interpolated on its own grid),
    # not a parent -- unlike law_of_motion.py's grid selection for the
    # *transition into* a child, which needs a representative parent. Grids live
    # on the state-choice space (that's where the solution itself lives), so
    # "choice" is a legitimate part of that identity, same as for
    # compute_utility/compute_marginal_utility below.
    continuous_state_space = compute_own_continuous_grid_combos(
        state_choice_vec, continuous_grid_functions, additional_continuous_state_names
    )

    # FUES-only (irregular) branch: check_model_config.py enforces at most one
    # additional continuous state here, so this is the only continuous state besides
    # assets_begin_of_period. Pass it under its actual state name.
    cont_state_name = list(continuous_state_space.keys())[0]

    def interp_on_single_wealth_point(wealth_point, second_cont_grid_point):

        policy_interp, value_interp = (
            interp2d_policy_and_value_on_wealth_and_regular_grid(
                continuous_state_space=continuous_state_space,
                wealth_grid=endog_grid_child_state_choice,
                policy_grid=policy_child_state_choice,
                value_grid=value_child_state_choice,
                wealth_point_to_interp=wealth_point,
                regular_point_to_interp=second_cont_grid_point,
                compute_utility=compute_utility,
                state_choice_vec=state_choice_vec,
                params=params,
                discount_factor=discount_factor,
            )
        )
        marg_util_interp = compute_marginal_utility(
            consumption=policy_interp,
            params=params,
            **state_choice_vec,
            **{cont_state_name: second_cont_grid_point},
        )

        return value_interp, marg_util_interp

    # Outer vmap applies first
    interp_over_single_wealth_and_income_shock_draw = vmap(
        vmap(
            vmap(
                interp_on_single_wealth_point,
                in_axes=(0, None),  # income shocks
            ),
            in_axes=(0, None),  # wealth grid
        ),
        in_axes=(0, 0),  # continuous state grid
    )
    # Old points: regular grid and endog grid
    # New points: continuous state next period and wealth next period
    value_interp, marg_util_interp = interp_over_single_wealth_and_income_shock_draw(
        assets_beginning_of_next_period, continuous_state_beginning_of_next_period
    )

    return value_interp, marg_util_interp
