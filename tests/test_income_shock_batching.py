"""The income-shock memory switch: blocking the draws must change nothing.

``model_config["income_shock_batch_size"]`` decides how many income-shock quadrature
draws the backward induction interpolates at once (see ``solve_single_period``). Left
out, it is ``n_quad_points``: one block holding every draw, which is the single-pass
solve. A smaller ``k`` processes the draws in blocks of ``k`` and adds each block's
weighted contribution to a running total, so the interpolated child arrays are never
built for more than ``k`` draws at a time. Both compute the same quadrature sum, so the
switch is a pure memory/parallelism knob and the solution must not move.

``k`` has to divide ``n_quad_points``, so the draws always split evenly. Checked across
every interpolation path the blocked step can take -- 1d wealth, 2d irregular (FUES with
an additional continuous state), n-d regular (Druedahl-Jorgensen) -- and with stochastic
transitions, for every divisor from one draw per block up to the single-block default.

"""

import jax.numpy as jnp
import numpy as np
import pytest

import dcegm
import dcegm.toy_models as toy_models

# =====================================================================================
# Model loaders -- one per interpolation path the blocked step can take
# =====================================================================================


def _retirement_1d():
    """No additional continuous state: 1d interpolation on wealth alone."""
    model_funcs = toy_models.load_example_model_functions("dcegm_paper")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_retirement_with_shocks"
        )
    )
    model_config = dict(model_config)
    model_config["n_periods"] = 6
    return model_funcs, params, model_specs, model_config


def _cont_exp_fues():
    """Additional continuous state with FUES: 2d irregular interpolation."""
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    return model_funcs, params, model_specs, dict(model_config)


def _cont_exp_dj():
    """Additional continuous state with Druedahl-Jorgensen: n-d regular."""
    model_funcs, params, model_specs, model_config = _cont_exp_fues()
    model_config["continuous_states"] = dict(model_config["continuous_states"])
    model_config["continuous_states"]["assets_begin_of_period"] = jnp.linspace(
        0, 50, 50
    )
    model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    return model_funcs, params, model_specs, model_config


def _stochastic_ltc_and_job_offer():
    """Stochastic transitions, so child states are integrated over as well."""
    model_funcs = toy_models.load_example_model_functions(
        "with_stochastic_ltc_and_job_offer"
    )
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "with_stochastic_ltc_and_job_offer"
        )
    )
    return model_funcs, params, model_specs, dict(model_config)


CASES = {
    "retirement_1d": _retirement_1d,
    "cont_exp_fues": _cont_exp_fues,
    "cont_exp_dj": _cont_exp_dj,
    "stochastic_ltc_and_job_offer": _stochastic_ltc_and_job_offer,
}

# The toy models come with 5 quadrature draws, whose only divisors are 1 and 5. Six
# draws give the block sizes an interior: 1 is the minimum-memory extreme, 2 and 3 run
# three and two blocks, 6 is the single-block default.
N_QUAD_POINTS = 6
BLOCK_SIZES = [1, 2, 3, 6]


def _solve_with(loader, income_shock_batch_size):
    model_funcs, params, model_specs, model_config = loader()
    model_config["n_quad_points"] = N_QUAD_POINTS
    model_config["income_shock_batch_size"] = income_shock_batch_size
    model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    return model.solve(params)


def _assert_same_solution(expected_solution, got_solution, block_size):
    for field in ("value", "policy", "endog_grid"):
        # endog_grid is None when the model does not store it (skip_endog_grid_storage).
        if getattr(expected_solution, field) is None:
            assert getattr(got_solution, field) is None
            continue
        expected = np.asarray(getattr(expected_solution, field))
        got = np.asarray(getattr(got_solution, field))
        np.testing.assert_array_equal(
            np.isfinite(expected),
            np.isfinite(got),
            err_msg=(
                f"{field}: NaN/inf layout differs between the one-pass solve and "
                f"income_shock_batch_size={block_size}"
            ),
        )
        finite = np.isfinite(expected)
        np.testing.assert_allclose(
            expected[finite],
            got[finite],
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"{field}: income_shock_batch_size={block_size} moved the solution",
        )


@pytest.mark.parametrize("case", CASES.keys())
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_blocked_shock_integration_matches_one_pass(case, block_size):
    loader = CASES[case]
    _assert_same_solution(
        _solve_with(loader, None), _solve_with(loader, block_size), block_size
    )


@pytest.mark.parametrize("case", CASES.keys())
def test_one_block_per_draw_set_is_the_default(case):
    """Asking for all draws in one block is exactly what leaving the key out does."""
    loader = CASES[case]
    default = _solve_with(loader, None)
    explicit = _solve_with(loader, N_QUAD_POINTS)

    for field in ("value", "policy", "endog_grid"):
        if getattr(default, field) is None:
            assert getattr(explicit, field) is None
            continue
        np.testing.assert_array_equal(
            np.asarray(getattr(default, field)), np.asarray(getattr(explicit, field))
        )


# =====================================================================================
# Config validation
# =====================================================================================


@pytest.mark.parametrize("bad_value", [0, -1, 2.0, True, "2", 7, 4])
def test_invalid_income_shock_batch_size_raises(bad_value):
    """Zero, negative, non-integer, more draws per block than there are, and 4 -- a
    positive integer in range that does not divide the six draws."""
    model_funcs, _, model_specs, model_config = _retirement_1d()
    model_config["n_quad_points"] = N_QUAD_POINTS
    model_config["income_shock_batch_size"] = bad_value
    with pytest.raises(ValueError, match="income_shock_batch_size"):
        dcegm.setup_model(
            model_config=model_config, model_specs=model_specs, **model_funcs
        )


def test_income_shock_batch_size_defaults_to_all_draws():
    model_funcs, _, model_specs, model_config = _retirement_1d()
    model_config.pop("income_shock_batch_size", None)
    model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    assert (
        model.model_config["income_shock_batch_size"]
        == model.model_config["n_quad_points"]
    )
