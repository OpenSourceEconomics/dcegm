"""The estimation likelihood must build and evaluate.

``create_experimental_ll_func`` is what every structural estimation run calls, and
it reaches a chain of choice-probability helpers that nothing else in the test suite
exercises -- ``create_partial_choice_prob_calculation`` ->
``calc_choice_prob_for_state_choices`` -> ``calc_choice_probs_for_states`` ->
``choice_values_for_states``. A signature change anywhere along it is invisible until
an estimation crashes, so these tests evaluate the likelihood end to end on a toy
model and check the contributions are finite and respond to the parameters.

"""

import copy

import jax.numpy as jnp
import numpy as np
import pytest

import dcegm
import dcegm.toy_models as toy_models

N_OBS = 24


@pytest.fixture(scope="module")
def model_params_and_data():
    model_funcs = toy_models.load_example_model_functions("dcegm_paper")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_retirement_with_shocks"
        )
    )
    model_config = copy.deepcopy(model_config)
    model_config["n_periods"] = 5
    model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )

    rng = np.random.default_rng(3141)
    observed_states = {
        "period": rng.integers(0, 4, N_OBS),
        # Working last period: retirement is absorbing, so lagged_choice = 1 leaves
        # only one choice and every contribution would be a degenerate -log(1).
        "lagged_choice": np.zeros(N_OBS, dtype=int),
        "dummy_stochastic": np.zeros(N_OBS, dtype=int),
        "assets_begin_of_period": rng.uniform(5.0, 40.0, N_OBS),
    }
    observed_choices = rng.integers(0, 2, N_OBS)
    return model, params, observed_states, observed_choices


def _build_ll_func(model, params, observed_states, observed_choices):
    return model.create_experimental_ll_func(
        params_all=params,
        observed_states=observed_states,
        observed_choices=observed_choices,
        return_model_solution=False,
    )


def test_likelihood_contributions_are_finite(model_params_and_data):
    model, params, observed_states, observed_choices = model_params_and_data
    ll_func = _build_ll_func(model, params, observed_states, observed_choices)

    contributions = np.asarray(ll_func(params))

    assert contributions.shape == (N_OBS,)
    assert np.all(np.isfinite(contributions))
    # Negative log-likelihood contributions of choices that are genuinely available:
    # strictly positive, since no choice is taken with probability one.
    assert np.all(contributions > 0)


def test_likelihood_responds_to_params(model_params_and_data):
    """A different taste-shock scale must move the contributions.

    Guards against the choice-probability chain quietly returning something that does
    not depend on the parameters at all.

    """
    model, params, observed_states, observed_choices = model_params_and_data
    ll_func = _build_ll_func(model, params, observed_states, observed_choices)

    at_start = np.asarray(ll_func(params))
    shifted = dict(params)
    shifted["taste_shock_scale"] = params["taste_shock_scale"] * 2
    at_shifted = np.asarray(ll_func(shifted))

    assert np.all(np.isfinite(at_shifted))
    assert not np.allclose(at_start, at_shifted)


def test_likelihood_can_return_the_solution(model_params_and_data):
    """``return_model_solution`` hands back the solve the likelihood just ran."""
    model, params, observed_states, observed_choices = model_params_and_data
    ll_func = model.create_experimental_ll_func(
        params_all=params,
        observed_states=observed_states,
        observed_choices=observed_choices,
        return_model_solution=True,
    )

    contributions, solution = ll_func(params)

    assert np.all(np.isfinite(np.asarray(contributions)))
    assert set(solution) == {"value", "policy", "endog_grid"}
    assert isinstance(solution["value"], jnp.ndarray)
