"""Regression test for a state-specific-grid bug in the upper-envelope refinement step.

`solve_euler_equation.py`'s EGM candidate generation already builds each state-choice's
own continuous-state combo grid on demand (via `compute_own_continuous_grid_combos`),
so the candidate policy/value are computed against the right grid. But
`solve_single_period.py`'s `run_upper_envelope` -- the FUES/DJ refinement step that runs
*after* those candidates exist -- used to feed the model-wide default grid into the
upper envelope's `value_function` (used for the constrained-region correction)
regardless of any per-state-choice override, instead of that state-choice's own grid.

This was invisible to every other test in this suite for two independent reasons, both
needed here to actually trigger it:

1. `with_cont_exp`'s CRRA utility function (`utility_crra(consumption, choice, params)`)
   does not take the continuous state ("experience") as an argument at all --
   `determine_function_arguments_and_partial_model_specs` filters it out before it
   reaches the function, so feeding the wrong value had zero effect on the output. This
   test uses a utility function that genuinely depends on "experience" directly (not
   just through consumption).
2. The FUES upper envelope (this toy model's default) only calls `value_function` (and
   therefore only reads `continuous_state_dict`) when the endogenous grid has a
   non-monotonicity overlapping the credit-constrained region -- a fairly special
   condition that this toy model/parameterization doesn't happen to trigger, so even
   with an experience-dependent utility, FUES alone would not have caught this. The
   Druedahl-Jorgensen upper envelope, by contrast, evaluates `value_function`
   unconditionally at every point on the common wealth grid (see
   `upper_envelope.jax.drued_jorg_jax`'s `v_all = vmap(_compute_value, ...)`), so it's
   the reliable path to exercise this bug.

Same pattern as test_state_specific_grid_end_to_end.py: a state-specific grid must
reproduce a model where that same grid is declared directly, bit-for-bit.

"""

import jax
import jax.numpy as jnp
import numpy as np

import dcegm
import dcegm.toy_models as toy_models
from dcegm.toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    inverse_marginal_utility_crra,
    marginal_utility_crra,
)


def _utility_crra_with_experience(consumption, choice, experience, params):
    utility_consumption = jax.lax.select(
        jnp.allclose(params["rho"], 1),
        jnp.log(consumption),
        (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"]),
    )
    # Additive term depending directly on the continuous state -- unlike the default
    # with_cont_exp utility, this makes the upper envelope's constrained-region
    # correction (which needs the right "experience" value) actually matter for the
    # output, not just for consumption/budget.
    return (
        utility_consumption
        - (1 - choice) * params["delta"]
        + params["exp"] * experience
    )


def _load_with_cont_exp_experience_dependent_utility():
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    model_funcs = dict(model_funcs)
    model_funcs["utility_functions"] = {
        "utility": _utility_crra_with_experience,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
    }
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    model_config = dict(model_config)
    model_config["continuous_states"] = dict(model_config["continuous_states"])
    model_config["continuous_states"]["assets_begin_of_period"] = jnp.linspace(
        0, 50, 50
    )
    model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    return model_funcs, params, model_specs, model_config


def test_state_specific_grid_matches_direct_declaration_with_experience_dependent_utility():
    model_funcs, params, model_specs, model_config = (
        _load_with_cont_exp_experience_dependent_utility()
    )

    scaled_grid = jnp.asarray(model_config["continuous_states"]["experience"]) * 2.0

    reference_model_config = dict(model_config)
    reference_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    reference_model_config["continuous_states"]["experience"] = scaled_grid
    reference_model = dcegm.setup_model(
        model_config=reference_model_config, model_specs=model_specs, **model_funcs
    )
    reference_solved = reference_model.solve(params)

    def constant_scaled_grid_func(period):
        return scaled_grid

    state_specific_model_config = dict(model_config)
    state_specific_model_config["continuous_states"] = dict(
        model_config["continuous_states"]
    )
    state_specific_model_config["continuous_states"]["experience"] = None
    state_specific_model = dcegm.setup_model(
        model_config=state_specific_model_config,
        model_specs=model_specs,
        continuous_grid_functions={"experience": constant_scaled_grid_func},
        **model_funcs,
    )
    state_specific_solved = state_specific_model.solve(params)

    np.testing.assert_array_equal(
        np.asarray(reference_solved.value), np.asarray(state_specific_solved.value)
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.policy), np.asarray(state_specific_solved.policy)
    )
    np.testing.assert_array_equal(
        np.asarray(reference_solved.endog_grid),
        np.asarray(state_specific_solved.endog_grid),
    )
