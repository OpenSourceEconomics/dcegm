"""Tests for state-specific continuous state grids (Phases 0 and 1).

Grids live on the state-choice space, not the bare state space -- that's where
the solution itself lives (value_solved/policy_solved/endog_grid_solved are
indexed by state-choice). A grid may therefore depend on the discrete state *and*
on "choice".

See docs/source/development/internals/state_specific_continuous_grids_plan.md.

"""

import numpy as np
import pytest

from dcegm.pre_processing.check_model_config import check_model_config_and_process
from dcegm.pre_processing.model_functions.process_model_functions import (
    process_continuous_grid_functions,
    process_sparsity_condition,
)
from dcegm.pre_processing.model_structure.continuous_state_grids import (
    check_continuous_grid_consistency_across_shared_children,
    evaluate_state_specific_continuous_grids,
)
from dcegm.pre_processing.model_structure.state_choice_space import (
    create_state_choice_space_and_child_state_mapping,
)
from dcegm.pre_processing.model_structure.state_space import create_state_space


def _base_model_config():
    return {
        "n_periods": 3,
        "n_quad_points": 1,
        "choices": [0, 1],
        "deterministic_states": {"group": [0, 1, 2]},
        "continuous_states": {
            "assets_end_of_period": np.linspace(0, 1, 10),
            "experience": np.linspace(0, 1, 4),
        },
    }


# =====================================================================================
# Phase 0: continuous_grid_functions plumbing
#
# continuous_grid_functions is a top-level argument to create_model_dict, following
# the same convention as shock_functions/stochastic_states_transitions -- functions
# are never nested inside model_config, which holds data/config only.
# =====================================================================================


def test_continuous_grid_functions_absent_defaults_to_constant_grids():
    # Both "assets_end_of_period" and "experience" go through the constant-wrapper
    # branch here (neither is state-specific), with two *different* default grids
    # -- a regression test for a late-binding-closure-in-a-loop bug where every
    # constant wrapper would have ended up returning the last-processed default
    # grid instead of its own (see process_continuous_grid_functions).
    processed = check_model_config_and_process(_base_model_config())
    continuous_grid_functions, state_specific_names = process_continuous_grid_functions(
        continuous_grid_functions=None, model_config=processed, model_specs={}
    )
    assert state_specific_names == []
    np.testing.assert_allclose(
        continuous_grid_functions["experience"](period=0, lagged_choice=0, group=1),
        np.linspace(0, 1, 4),
    )
    np.testing.assert_allclose(
        continuous_grid_functions["assets_end_of_period"](
            period=0, lagged_choice=0, group=1
        ),
        np.linspace(0, 1, 10),
    )


def test_continuous_grid_functions_must_be_a_dict():
    processed = check_model_config_and_process(_base_model_config())
    with pytest.raises(ValueError, match="must be a dictionary"):
        process_continuous_grid_functions(
            continuous_grid_functions=[1, 2], model_config=processed, model_specs={}
        )


def test_continuous_grid_functions_rejects_unknown_state_name():
    processed = check_model_config_and_process(_base_model_config())
    with pytest.raises(ValueError, match="not a continuous state"):
        process_continuous_grid_functions(
            continuous_grid_functions={"not_a_state": lambda group: group},
            model_config=processed,
            model_specs={},
        )


def test_continuous_grid_functions_rejects_non_callable():
    processed = check_model_config_and_process(_base_model_config())
    with pytest.raises(ValueError, match="must be a callable"):
        process_continuous_grid_functions(
            continuous_grid_functions={"experience": np.zeros(4)},
            model_config=processed,
            model_specs={},
        )


def test_process_continuous_grid_functions_wraps_state_specific_and_default_names():
    def grid_func(group):
        return np.linspace(0, 1, 4) * (group + 1)

    processed = check_model_config_and_process(_base_model_config())
    continuous_grid_functions, state_specific_names = process_continuous_grid_functions(
        continuous_grid_functions={"experience": grid_func},
        model_config=processed,
        model_specs={},
    )
    assert state_specific_names == ["experience"]

    # State-specific name: varies with the discrete state.
    np.testing.assert_allclose(
        continuous_grid_functions["experience"](
            period=0, lagged_choice=0, group=1, choice=0
        ),
        np.linspace(0, 1, 4) * 2,
    )
    np.testing.assert_allclose(
        continuous_grid_functions["experience"](
            period=0, lagged_choice=0, group=2, choice=0
        ),
        np.linspace(0, 1, 4) * 3,
    )

    # Default name: constant regardless of the discrete state passed in.
    np.testing.assert_allclose(
        continuous_grid_functions["assets_end_of_period"](
            period=0, lagged_choice=0, group=0, choice=0
        ),
        np.linspace(0, 1, 10),
    )
    np.testing.assert_allclose(
        continuous_grid_functions["assets_end_of_period"](
            period=2, lagged_choice=1, group=2, choice=1
        ),
        np.linspace(0, 1, 10),
    )


def test_process_continuous_grid_functions_allows_choice_dependence():
    # Grids live on the state-choice space, so "choice" is a legitimate parameter,
    # unlike model_specs/state-only functions elsewhere in the codebase.
    def grid_func(choice):
        return np.linspace(0, 1, 4) * (choice + 1)

    processed = check_model_config_and_process(_base_model_config())
    continuous_grid_functions, state_specific_names = process_continuous_grid_functions(
        continuous_grid_functions={"experience": grid_func},
        model_config=processed,
        model_specs={},
    )
    assert state_specific_names == ["experience"]
    np.testing.assert_allclose(
        continuous_grid_functions["experience"](
            period=0, lagged_choice=0, group=1, choice=1
        ),
        np.linspace(0, 1, 4) * 2,
    )


def test_evaluate_state_specific_grids_rejects_wrong_length():
    state_choice_space = np.array([[0, 0, 0, 0], [0, 0, 1, 1]])
    discrete_state_choice_names = ["period", "lagged_choice", "group", "choice"]
    continuous_states_info = {
        "additional_continuous_state_grids": {"experience": np.linspace(0, 1, 4)},
        "assets_grid_end_of_period": np.linspace(0, 1, 10),
    }

    def bad_grid_func(**kwargs):
        return np.linspace(0, 1, 3) if kwargs["group"] == 0 else np.linspace(0, 1, 4)

    with pytest.raises(ValueError, match="must be a 1d array"):
        evaluate_state_specific_continuous_grids(
            state_choice_space=state_choice_space,
            discrete_state_choice_names=discrete_state_choice_names,
            continuous_grid_functions={"experience": bad_grid_func},
            state_specific_names=["experience"],
            continuous_states_info=continuous_states_info,
        )


def test_evaluate_state_specific_grids_skips_names_without_user_callable():
    state_choice_space = np.array([[0, 0, 0, 0], [0, 0, 1, 1]])
    discrete_state_choice_names = ["period", "lagged_choice", "group", "choice"]
    continuous_states_info = {
        "additional_continuous_state_grids": {"experience": np.linspace(0, 1, 4)},
        "assets_grid_end_of_period": np.linspace(0, 1, 10),
    }

    grids_per_state_choice = evaluate_state_specific_continuous_grids(
        state_choice_space=state_choice_space,
        discrete_state_choice_names=discrete_state_choice_names,
        continuous_grid_functions={},
        state_specific_names=[],
        continuous_states_info=continuous_states_info,
    )
    assert grids_per_state_choice == {}


# =====================================================================================
# Phase 1: consistency check across state-choices sharing a child state
# =====================================================================================


def _build_state_choice_objects(grid_func_dict, group_resets_on_choice=False):
    processed_config = check_model_config_and_process(_base_model_config())

    continuous_grid_functions, state_specific_names = process_continuous_grid_functions(
        continuous_grid_functions=grid_func_dict,
        model_config=processed_config,
        model_specs={},
    )

    sparsity_condition = process_sparsity_condition(
        state_space_functions={}, model_specs={}
    )
    state_space_objects = create_state_space(
        model_config=processed_config, sparsity_condition=sparsity_condition
    )

    def next_period_deterministic_state(**kwargs):
        # By default "group" is time invariant (passes through unchanged from
        # parent to child), like sex/education would be. When
        # group_resets_on_choice is True, it instead becomes a function of the
        # choice -- i.e. it no longer passes through 1:1, so two different parents
        # can share a child while disagreeing on their own "group". "choice"
        # itself always passes through 1:1 into the child's own "lagged_choice"
        # (enforced by check_endog_update_function in state_choice_space.py), so a
        # grid depending only on "choice" can never violate the consistency check,
        # regardless of what else is going on.
        next_group = kwargs["choice"] % 3 if group_resets_on_choice else kwargs["group"]
        return {
            "period": kwargs["period"] + 1,
            "lagged_choice": kwargs["choice"],
            "group": next_group,
        }

    def state_specific_choice_set(**kwargs):
        return np.array([0, 1])

    return (
        processed_config,
        state_space_objects,
        next_period_deterministic_state,
        state_specific_choice_set,
        continuous_grid_functions,
        state_specific_names,
    )


def test_grid_depending_on_time_invariant_state_passes():
    def grid_func(group):
        return np.linspace(0, 1, 4) * (group + 1)

    (
        processed_config,
        state_space_objects,
        next_period_deterministic_state,
        state_specific_choice_set,
        continuous_grid_functions,
        state_specific_names,
    ) = _build_state_choice_objects({"experience": grid_func})

    # "group" is time invariant and shared by parent and child by construction, so
    # any two parents sharing a child necessarily agree on it -- this must not raise.
    create_state_choice_space_and_child_state_mapping(
        model_config=processed_config,
        state_specific_choice_set=state_specific_choice_set,
        next_period_deterministic_state=next_period_deterministic_state,
        state_space_arrays=state_space_objects,
        continuous_grid_functions=continuous_grid_functions,
        state_specific_continuous_grid_names=state_specific_names,
    )


def test_grid_depending_on_choice_always_passes():
    # "choice" always passes through 1:1 into the child (it becomes the child's
    # own "lagged_choice") -- so two parents sharing a child are guaranteed to
    # have made the same choice, and a grid depending only on "choice" can never
    # violate the consistency check. Combined with group_resets_on_choice=True,
    # which *would* break a group-dependent grid (see the test below), to show
    # this holds regardless of what else varies.
    def grid_func(choice):
        return np.linspace(0, 1, 4) * (choice + 1)

    (
        processed_config,
        state_space_objects,
        next_period_deterministic_state,
        state_specific_choice_set,
        continuous_grid_functions,
        state_specific_names,
    ) = _build_state_choice_objects(
        {"experience": grid_func}, group_resets_on_choice=True
    )

    create_state_choice_space_and_child_state_mapping(
        model_config=processed_config,
        state_specific_choice_set=state_specific_choice_set,
        next_period_deterministic_state=next_period_deterministic_state,
        state_space_arrays=state_space_objects,
        continuous_grid_functions=continuous_grid_functions,
        state_specific_continuous_grid_names=state_specific_names,
    )


def test_grid_depending_on_lagged_choice_fails():
    def grid_func(lagged_choice):
        return np.linspace(0, 1, 4) * (lagged_choice + 1)

    (
        processed_config,
        state_space_objects,
        next_period_deterministic_state,
        state_specific_choice_set,
        continuous_grid_functions,
        state_specific_names,
    ) = _build_state_choice_objects({"experience": grid_func})

    # A parent with lagged_choice=0 and one with lagged_choice=1 both choosing
    # choice=1 land on the same child (period+1, lagged_choice=1, same group), but
    # disagree on their own grid (indexed by their own lagged_choice -- note this
    # is the *parent's* pre-existing state, not "choice" itself, which always
    # passes through consistently) -- must raise.
    with pytest.raises(ValueError, match="different grids"):
        create_state_choice_space_and_child_state_mapping(
            model_config=processed_config,
            state_specific_choice_set=state_specific_choice_set,
            next_period_deterministic_state=next_period_deterministic_state,
            state_space_arrays=state_space_objects,
            continuous_grid_functions=continuous_grid_functions,
            state_specific_continuous_grid_names=state_specific_names,
        )


def test_grid_on_state_that_does_not_pass_through_1to1_fails():
    def grid_func(group):
        return np.linspace(0, 1, 4) * (group + 1)

    (
        processed_config,
        state_space_objects,
        next_period_deterministic_state,
        state_specific_choice_set,
        continuous_grid_functions,
        state_specific_names,
    ) = _build_state_choice_objects(
        {"experience": grid_func}, group_resets_on_choice=True
    )

    # Now "group" is overwritten by (choice % 3) in the child, so two parents with
    # different own "group" but the same choice share a child while disagreeing on
    # their grid (indexed by their own "group") -- must raise.
    with pytest.raises(ValueError, match="different grids"):
        create_state_choice_space_and_child_state_mapping(
            model_config=processed_config,
            state_specific_choice_set=state_specific_choice_set,
            next_period_deterministic_state=next_period_deterministic_state,
            state_space_arrays=state_space_objects,
            continuous_grid_functions=continuous_grid_functions,
            state_specific_continuous_grid_names=state_specific_names,
        )


def test_no_state_specific_grids_is_a_no_op():
    (
        processed_config,
        state_space_objects,
        next_period_deterministic_state,
        state_specific_choice_set,
        _,
        _,
    ) = _build_state_choice_objects({})

    create_state_choice_space_and_child_state_mapping(
        model_config=processed_config,
        state_specific_choice_set=state_specific_choice_set,
        next_period_deterministic_state=next_period_deterministic_state,
        state_space_arrays=state_space_objects,
    )


def test_check_function_directly_with_hand_built_mapping():
    # Two state-choice rows (0 and 1) share child state index 5, and disagree on
    # their own grid for "experience" -- the check should name both rows.
    state_choice_space = np.array(
        [
            [0, 0, 0, 0],  # period, lagged_choice, group, choice
            [0, 1, 0, 0],
        ]
    )
    discrete_states_names = ["period", "lagged_choice", "group"]
    map_state_choice_to_child_states = np.array([[5], [5]])
    grids_per_state_choice = {
        "experience": np.array(
            [
                [0.0, 1.0],  # row 0's own grid
                [0.0, 2.0],  # row 1's own grid -- disagrees
            ]
        )
    }

    with pytest.raises(ValueError, match="child state index 5"):
        check_continuous_grid_consistency_across_shared_children(
            state_choice_space=state_choice_space,
            discrete_states_names=discrete_states_names,
            map_state_choice_to_child_states=map_state_choice_to_child_states,
            grids_per_state_choice=grids_per_state_choice,
        )


def test_check_function_directly_passes_when_grids_agree():
    state_choice_space = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    discrete_states_names = ["period", "lagged_choice", "group"]
    map_state_choice_to_child_states = np.array([[5], [5]])
    grids_per_state_choice = {
        "experience": np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
    }

    check_continuous_grid_consistency_across_shared_children(
        state_choice_space=state_choice_space,
        discrete_states_names=discrete_states_names,
        map_state_choice_to_child_states=map_state_choice_to_child_states,
        grids_per_state_choice=grids_per_state_choice,
    )
