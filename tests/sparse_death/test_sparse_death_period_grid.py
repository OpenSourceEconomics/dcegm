"""A period-specific second-continuous grid builds and solves even though every death
state proxies to a single last-period death slot.

The grid-consistency check groups by the actual child (each its own), so the proxy-
collapsed death children do not force a shared grid. The solve is decoupled from the
proxy too: the law of motion runs on the non-proxy child (its real age), and only the
value/policy lookup follows the proxy to its solved slot (see ``calc_law_of_motion`` /
``child_state_dedup.py``).

The oracle at the bottom solves the model with death proxied and with death solved
explicitly per age and compares the living-state values: they agree even when the
bequest depends on experience, which validates the proxy value redirect.

"""

import jax.numpy as jnp
import numpy as np
import pytest

import dcegm
from tests.sparse_death.budget import budget_constraint_exp
from tests.sparse_death.state_space import create_state_space_functions
from tests.sparse_death.stochastic_processes import job_offer, prob_survival
from tests.sparse_death.utility import (
    create_final_period_utility_function_dict,
    create_utility_function_dict,
)

N_PERIODS = 20
N_EXP = 7

_MODEL_SPECS = {
    "n_periods": N_PERIODS,
    "n_choices": 3,
    "min_ret_period": 5,
    "max_ret_period": 10,
    "fresh_bonus": 0.1,
    "exp_scale": 20,
}
_PARAMS = {
    "delta": 0.5,
    "discount_factor": 0.95,
    "taste_shock_scale": 1,
    "income_shock_std": 1,
    "income_shock_mean": 0.0,
    "interest_rate": 0.05,
    "constant": 1,
    "exp": 0.1,
    "exp_squared": -0.01,
    "consumption_floor": 0.5,
}


def _base_config():
    return {
        "min_period_batch_segments": [5, 12],
        "n_periods": N_PERIODS,
        "choices": np.arange(3, dtype=int),
        "deterministic_states": {"already_retired": np.arange(2, dtype=int)},
        "continuous_states": {
            "assets_end_of_period": jnp.arange(0, 100, 5, dtype=float),
            "experience": jnp.linspace(0, 1, N_EXP, dtype=float),
        },
        "stochastic_states": {"job_offer": [0, 1], "survival": [0, 1]},
        "n_quad_points": 5,
    }


def _setup(
    model_config,
    continuous_grid_functions=None,
    utility_functions=None,
    utility_functions_final_period=None,
    state_space_functions=None,
):
    return dcegm.setup_model(
        model_specs=_MODEL_SPECS,
        model_config=model_config,
        utility_functions=utility_functions or create_utility_function_dict(),
        utility_functions_final_period=(
            utility_functions_final_period
            or create_final_period_utility_function_dict()
        ),
        state_space_functions=state_space_functions or create_state_space_functions(),
        budget_constraint=budget_constraint_exp,
        stochastic_states_transitions={
            "job_offer": job_offer,
            "survival": prob_survival,
        },
        continuous_grid_functions=continuous_grid_functions,
    )


def _period_specific_exp_grid(period):
    # A genuinely period-dependent grid, spanning [0, 0.9] .. [0, 1.0] over the life
    # cycle, so each age's death child keeps its own grid.
    return jnp.linspace(0, 1, N_EXP) * (0.9 + 0.1 * period / (N_PERIODS - 1))


def _constant_exp_grid(period):
    # Ignores period: same grid as the declared-array baseline.
    return jnp.linspace(0, 1, N_EXP)


def test_period_specific_grid_builds_and_solves():
    config = _base_config()
    config["continuous_states"]["experience"] = None  # supplied by the grid function
    model = _setup(config, {"experience": _period_specific_exp_grid})
    solved = model.solve(params=_PARAMS)
    assert np.isfinite(
        np.asarray(solved.value)[~np.isnan(np.asarray(solved.value))]
    ).all()


def test_constant_grid_function_matches_declared_array():
    # Sanity/no-regression: a period-*independent* grid supplied via
    # continuous_grid_functions must reproduce the declared-array solve.
    baseline = _setup(_base_config()).solve(params=_PARAMS)

    config = _base_config()
    config["continuous_states"]["experience"] = None
    via_func = _setup(config, {"experience": _constant_exp_grid}).solve(params=_PARAMS)

    for name in ("value", "policy", "endog_grid"):
        a = np.asarray(getattr(baseline, name))
        b = np.asarray(getattr(via_func, name))
        m = np.isfinite(a) & np.isfinite(b)
        # Relative: ``value`` differs up to ~1e7 across machines, where it hits ~1e22.
        assert np.allclose(a[m], b[m], rtol=1e-10, atol=1e-10), name


def _bequest_utility_dict(bequest_slope):
    """Utility whose death (survival == 0) branch is bequest = log(wealth) +
    bequest_slope * experience -- a death value that depends on the second
    continuous state. Marginal utility of consumption is unchanged (1 / c)."""

    def utility(consumption, survival, choice, experience, params):
        alive = jnp.log(consumption) - choice * params["delta"]
        dead = jnp.log(consumption) + bequest_slope * experience
        return jnp.where(survival == 0, dead, alive)

    return {
        "utility": utility,
        "marginal_utility": lambda consumption, survival, choice, experience, params: 1
        / consumption,
        "inverse_marginal_utility": lambda marginal_utility: 1 / marginal_utility,
    }


def _bequest_final_utility_dict(bequest_slope):
    return {
        "utility": lambda wealth, experience, params: jnp.log(wealth)
        + bequest_slope * experience,
        "marginal_utility": lambda wealth, experience, params: 1 / wealth,
    }


def _iter_batch_infos(batch_info):
    for key, seg in batch_info.items():
        if not key.startswith("batches_info_segment_"):
            continue
        yield seg
        if not seg.get("batches_cover_all", True):
            yield seg["last_batch_info"]


def test_experience_dependent_bequest_solves_and_redirects_through_proxy():
    # A bequest that depends on experience solves with death proxied to one
    # last-period slot, and the proxy structure holds: the value lookup collapses
    # every death age onto that slot while the law-of-motion transition child keeps
    # the child's real age.
    config = _base_config()
    config["continuous_states"]["experience"] = None
    model = _setup(
        config,
        {"experience": _period_specific_exp_grid},
        utility_functions=_bequest_utility_dict(0.3),
        utility_functions_final_period=_bequest_final_utility_dict(0.3),
    )
    solved = model.solve(params=_PARAMS)
    value = np.asarray(solved.value)
    assert np.isfinite(value[~np.isnan(value)]).all()

    last_period = N_PERIODS - 1
    found_earlier_transition_age = False
    for seg in _iter_batch_infos(model.batch_info):
        proxy = seg["state_choices_childs"]
        lma = seg["law_of_motion_arrays"]
        dead = np.asarray(proxy["survival"]) == 0
        # The value/policy this row reads lives in the reduced space: every death
        # child is redirected to a last-period death slot.
        assert np.all(np.asarray(proxy["period"])[dead] == last_period)
        # The transition, by contrast, runs on the child's real (non-proxy) age.
        transition_period = np.take_along_axis(
            np.asarray(lma["unique_child_states"]["period"]),
            np.asarray(lma["state_row_for_state_choice"]),
            axis=-1,
        )
        found_earlier_transition_age |= bool(
            np.any(transition_period[dead] < last_period)
        )
    assert found_earlier_transition_age
