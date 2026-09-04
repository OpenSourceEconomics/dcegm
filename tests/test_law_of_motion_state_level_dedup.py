"""Tests for the state-level dedup fast path in the law of motion.

The transition into a child does not depend on the child's own future choice --
``calc_law_of_motion_for_state_choices`` pops ``"choice"`` before calling the user's
functions. So whenever the user's budget/continuous-state functions don't declare
``choice`` either, every state-choice sharing a child state would compute a bit-
identical transition, and the solve instead evaluates it once per unique child *state*
and gathers the result out (see ``calc_law_of_motion_for_child_states`` in
law_of_motion.py, and ``_transition_funcs_depend_on_choice`` in
process_model_functions.py).

That is purely a cost optimization, so the central test here is an equivalence check:
forcing the slow (per-state-choice) path by adding an unused ``choice`` argument to the
budget equation must reproduce the fast path bit-for-bit.

"""

import jax.numpy as jnp
import numpy as np

import dcegm
import dcegm.toy_models as toy_models
from dcegm.toy_models.cons_ret_model_with_cont_exp.budget_constraint import (
    budget_constraint_cont_exp,
)


def _load_with_cont_exp():
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    return model_funcs, params, model_specs, model_config


def _budget_constraint_with_unused_choice(
    period,
    lagged_choice,
    choice,
    experience,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
    model_specs,
):
    """Identical economics to budget_constraint_cont_exp, but declaring ``choice``.

    ``choice`` is deliberately unused in the body: it only changes which granularity the
    law of motion is evaluated at (per state-choice rather than per unique child state),
    which must not change the result.

    """
    return budget_constraint_cont_exp(
        period=period,
        lagged_choice=lagged_choice,
        experience=experience,
        asset_end_of_previous_period=asset_end_of_previous_period,
        income_shock_previous_period=income_shock_previous_period,
        params=params,
        model_specs=model_specs,
    )


def test_default_toy_models_take_the_state_level_fast_path():
    # None of the shipped toy models' transition functions declare "choice", so
    # they must all take the deduplicated path -- otherwise the rest of the test
    # suite would never exercise it.
    for name in ["with_cont_exp", "with_exp", "dcegm_paper"]:
        model_funcs = toy_models.load_example_model_functions(name)
        params, model_specs, model_config = (
            toy_models.load_example_params_model_specs_and_config(name)
        )
        model = dcegm.setup_model(
            model_config=model_config, model_specs=model_specs, **model_funcs
        )
        assert not model.model_funcs["transition_funcs_depend_on_choice"], name


def test_choice_in_budget_signature_selects_the_state_choice_path():
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()
    model_funcs = dict(model_funcs)
    model_funcs["budget_constraint"] = _budget_constraint_with_unused_choice

    model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    assert model.model_funcs["transition_funcs_depend_on_choice"]


def test_state_level_and_state_choice_level_paths_agree_bit_for_bit():
    # The core equivalence check: a budget equation that declares (but ignores)
    # "choice" forces the per-state-choice path, and must reproduce the
    # deduplicated per-state path exactly. A dedup/gather bug -- e.g. a wrong
    # state_row_for_state_choice mapping -- would show up here as mismatched
    # value/policy, since it would feed each child the wrong state's transition.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    fast_path_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    assert not fast_path_model.model_funcs["transition_funcs_depend_on_choice"]
    fast_path_solved = fast_path_model.solve(params)

    slow_path_funcs = dict(model_funcs)
    slow_path_funcs["budget_constraint"] = _budget_constraint_with_unused_choice
    slow_path_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **slow_path_funcs
    )
    assert slow_path_model.model_funcs["transition_funcs_depend_on_choice"]
    slow_path_solved = slow_path_model.solve(params)

    np.testing.assert_array_equal(
        np.asarray(fast_path_solved.value), np.asarray(slow_path_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(fast_path_solved.policy), np.asarray(slow_path_solved.policy)
    )
    np.testing.assert_array_equal(
        np.asarray(fast_path_solved.endog_grid),
        np.asarray(slow_path_solved.endog_grid),
    )


def test_genuinely_choice_dependent_budget_differs_from_choice_free_one():
    # Sensitivity check for the test above: confirms a budget equation that
    # actually *uses* choice produces a different solution, so the bit-for-bit
    # agreement there reflects the two paths genuinely computing the same thing,
    # not "choice" being unable to affect the budget at all.
    model_funcs, params, model_specs, model_config = _load_with_cont_exp()

    def budget_constraint_using_choice(
        period,
        lagged_choice,
        choice,
        experience,
        asset_end_of_previous_period,
        income_shock_previous_period,
        params,
        model_specs,
    ):
        base = budget_constraint_cont_exp(
            period=period,
            lagged_choice=lagged_choice,
            experience=experience,
            asset_end_of_previous_period=asset_end_of_previous_period,
            income_shock_previous_period=income_shock_previous_period,
            params=params,
            model_specs=model_specs,
        )
        return base + 0.5 * choice

    baseline_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    baseline_solved = baseline_model.solve(params)

    choice_dependent_funcs = dict(model_funcs)
    choice_dependent_funcs["budget_constraint"] = budget_constraint_using_choice
    choice_dependent_model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **choice_dependent_funcs
    )
    choice_dependent_solved = choice_dependent_model.solve(params)

    baseline_value = np.asarray(baseline_solved.value)
    choice_dependent_value = np.asarray(choice_dependent_solved.value)
    finite = np.isfinite(baseline_value) & np.isfinite(choice_dependent_value)
    assert not np.allclose(baseline_value[finite], choice_dependent_value[finite])
