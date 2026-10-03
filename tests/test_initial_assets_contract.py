"""The first period's wealth comes from the budget equation, like every other period.

``states_initial`` carries ``assets_end_of_previous_period``: what agents bring into the
first period, not what they have to spend in it. These tests pin that down -- that
period 0's ``assets_begin_of_period`` is exactly what the model's own budget equation
produces from the supplied assets and period 0's own income shock, that the shock is the
one reported on period 0's own row, and that the old key is rejected rather than
silently reinterpreted.

"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dcegm
import dcegm.toy_models as toy_models

N_AGENTS = 50
SEED = 4321


@pytest.fixture(scope="module")
def solved_paper_model():
    model_funcs = toy_models.load_example_model_functions("dcegm_paper")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("dcegm_paper")
    )
    model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    return model, model.solve(params), params, model_specs, model_config


def _initial_states(assets_end_of_previous_period):
    return {
        "period": np.zeros(N_AGENTS, dtype=int),
        "lagged_choice": np.zeros(N_AGENTS, dtype=int),
        "assets_end_of_previous_period": assets_end_of_previous_period,
    }


def test_first_period_wealth_is_the_budget_equation_applied_to_the_given_assets(
    solved_paper_model,
):
    _model, solved, params, model_specs, _model_config = solved_paper_model
    assets_end_of_previous_period = np.linspace(1.0, 20.0, N_AGENTS)

    df = solved.simulate(
        states_initial=_initial_states(assets_end_of_previous_period), seed=SEED
    )

    # Each period draws its own income shock and reports it on its own row, so
    # period 0's wealth can be reconciled straight from period 0's output.
    income_shock = df.xs(0, level="period")["income_shock"].to_numpy()

    budget_constraint = toy_models.load_example_model_functions("dcegm_paper")[
        "budget_constraint"
    ]
    expected = np.array(
        [
            budget_constraint(
                period=0,
                lagged_choice=0,
                asset_end_of_previous_period=assets_end_of_previous_period[i],
                income_shock_previous_period=income_shock[i],
                model_specs=model_specs,
                params=params,
            )
            for i in range(N_AGENTS)
        ]
    )

    simulated = df.xs(0, level="period")["assets_begin_of_period"].to_numpy()
    assert_allclose(simulated, expected, rtol=1e-6)

    # Guards the test itself: the budget equation must actually move the number, or
    # the assertion above would also pass if the supplied assets were used directly.
    assert not np.allclose(expected, assets_end_of_previous_period)


def test_more_assets_carried_in_means_more_wealth_in_the_first_period(
    solved_paper_model,
):
    """The supplied assets are an input to period 0, not an unused placeholder."""
    _model, solved, _params, _model_specs, _model_config = solved_paper_model

    df_poor = solved.simulate(
        states_initial=_initial_states(np.full(N_AGENTS, 5.0)), seed=SEED
    )
    df_rich = solved.simulate(
        states_initial=_initial_states(np.full(N_AGENTS, 25.0)), seed=SEED
    )

    assets_poor = df_poor.xs(0, level="period")["assets_begin_of_period"].to_numpy()
    assets_rich = df_rich.xs(0, level="period")["assets_begin_of_period"].to_numpy()
    assert np.all(assets_rich > assets_poor)


def test_old_initial_assets_key_is_rejected(solved_paper_model):
    """``assets_begin_of_period`` meant something different, so it must not be read."""
    _model, solved, _params, _model_specs, _model_config = solved_paper_model

    states_initial = _initial_states(np.full(N_AGENTS, 10.0))
    states_initial["assets_begin_of_period"] = states_initial.pop(
        "assets_end_of_previous_period"
    )

    with pytest.raises(ValueError, match="assets_end_of_previous_period"):
        solved.simulate(states_initial=states_initial, seed=SEED)


def test_missing_initial_assets_key_is_rejected(solved_paper_model):
    _model, solved, _params, _model_specs, _model_config = solved_paper_model

    states_initial = _initial_states(np.full(N_AGENTS, 10.0))
    del states_initial["assets_end_of_previous_period"]

    with pytest.raises(ValueError, match="assets_end_of_previous_period"):
        solved.simulate(states_initial=states_initial, seed=SEED)


def test_each_period_reports_the_income_shock_that_built_its_own_wealth(
    solved_paper_model,
):
    """One row of the output is enough to reconcile that row's wealth.

    Each period draws its own income shock, feeds it to its own budget equation and
    reports it on its own row. So for every period -- not just the first -- the budget
    equation applied to the previous row's savings and this row's ``income_shock`` must
    reproduce this row's ``assets_begin_of_period``.

    """
    _model, solved, params, model_specs, _model_config = solved_paper_model
    assets_end_of_previous_period = np.linspace(1.0, 20.0, N_AGENTS)

    df = solved.simulate(
        states_initial=_initial_states(assets_end_of_previous_period), seed=SEED
    )
    budget_constraint = toy_models.load_example_model_functions("dcegm_paper")[
        "budget_constraint"
    ]

    periods = sorted(df.index.get_level_values("period").unique())
    for period in periods:
        this = df.xs(period, level="period")
        carried_in = (
            assets_end_of_previous_period
            if period == periods[0]
            else df.xs(period - 1, level="period")["savings"].to_numpy()
        )
        expected = np.array(
            [
                budget_constraint(
                    period=period,
                    lagged_choice=int(this["lagged_choice"].to_numpy()[i]),
                    asset_end_of_previous_period=carried_in[i],
                    income_shock_previous_period=this["income_shock"].to_numpy()[i],
                    model_specs=model_specs,
                    params=params,
                )
                for i in range(N_AGENTS)
            ]
        )
        assert_allclose(
            this["assets_begin_of_period"].to_numpy(),
            expected,
            rtol=1e-6,
            err_msg=f"period {period}",
        )

    # The final period is included above only if it reports a real shock rather than
    # a placeholder; guard that it does.
    assert df.xs(periods[-1], level="period")["income_shock"].std() > 0
