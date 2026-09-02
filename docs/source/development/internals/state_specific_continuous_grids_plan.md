# State-specific continuous state grids — implementation plan

> **Note.** This page is AI-drafted (with Claude, based on reading the source and a
> design discussion with a maintainer) and is a *plan*, updated as phases land and
> design decisions change — see the Status section immediately below for what is
> actually shipped on the `state_grids` branch right now versus what is still
> outstanding. It is meant as a working reference for whoever implements this
> feature (human or AI). See also [`batching.rst`](batching.rst), which documents
> the mechanism this feature modifies (deduplication of child states across parent
> state-choices).

## Status (branch `state_grids`)

**Done, tested, passing the full `dcegm` suite:** all phases, 0 through 7. Phases
2/4/5 shipped as one combined change — see the note under Phase 2 for why they
turned out not to be separable. This feature is complete.

**Naming note:** despite the feature's name, grids ended up living on the
*state-choice* space, not the bare state space — see item 4 below and the Config
API section. "State-specific" in file/variable names predating that correction
(e.g. some docstrings, the doc title itself) should be read as "state-choice
specific."

Four real bugs/design errors were found and fixed during implementation. None of
them were caught by the tests written alongside the change that introduced them —
each was either caught by a maintainer reading the diff, or by a *stronger* test
written afterward once the weakness of the first test was noticed. Worth
remembering when judging how much to trust the tests below; "the tests pass" has
undersold correctness multiple times in this feature:

1. Grid functions were first added as a key inside `model_config`. Wrong — every
   other user-supplied function in `dcegm` is a separate top-level argument to
   `create_model_dict`, never nested in `model_config` (data/config only). Fixed:
   `continuous_grid_functions` is now a top-level argument, same convention as
   `shock_functions`. See the Config API section below. (Caught by a maintainer.)
2. The law-of-motion grid-selection code used the **child's** own discrete
   identity to pick which grid to feed the transition function, when it must be a
   **representative parent's** identity. The child's identity is correct for the
   transition function call itself (its `lagged_choice` etc. genuinely determine
   the transition), but the grid *values* fed into that call must be the parent's
   own grid — the whole reason Phase 1's consistency check exists. This only
   showed up once real batching/dedup was exercised; isolated unit tests that
   construct their own discrete-state dicts by hand didn't catch it. Fixed by
   threading a representative-parent state index through batch creation (folded
   Phase 4 into Phase 2 as a result) and through the last-two-periods special case
   (folded into Phase 5). (Caught by a maintainer.)
3. `final_periods.py`'s own final-period value computation
   (`calc_value_and_budget_for_each_gridpoint`) still read
   `model_structure["continuous_state_space"]` (global) for its own combo axis —
   the final-period analog of what `solve_euler_equation.py` needed fixing for
   every other period, and easy to miss because it's a separate code path with no
   shared function to inherit the fix from. Caught by
   `test_constant_but_different_grid_matches_direct_model_config_declaration`
   (see Phase 3 below) — a materially stronger end-to-end test than the ones
   written earlier in this feature, which only checked "reproduces the *default*
   grid" or "produces *different, finite* numbers." Those two weaker checks both
   still passed with this bug present; only a test that could distinguish "used
   the right state-specific grid" from "silently fell back to the old global
   grid" caught it. Worth designing new correctness tests this way going forward,
   not the weaker style used for Phase 2/4/5.
4. The entire design was scoped around the bare *discrete state*, matching the
   feature's name -- Phase 1's check evaluated `grid_func` once per row of
   `state_space`, and every "own grid" selection collapsed a state-choice down to
   its bare state before calling `grid_func`. Wrong: the solution itself
   (`value_solved`/`policy_solved`/`endog_grid_solved`) is indexed by
   *state-choice*, not state (`n_state_choices = state_choice_space.shape[0]` in
   `create_solution_container`), so that's the natural granularity for a grid too
   -- and there's a real, common use case for it (e.g. a choice like "retire"
   needing a narrower experience range than "work"). The first attempt at
   reconciling this went the wrong direction: rather than extending the design
   to state-choice, it tried to explicitly *forbid* `grid_func` from depending on
   `"choice"` (to make the existing state-only Phase 1 check sound). Corrected by
   moving the whole mechanism to the state-choice space instead -- `evaluate_state_specific_continuous_grids`
   now evaluates once per row of `state_choice_space`, and the collapsing-to-bare-state
   step was removed from `child_state_dedup.py` and `last_two_periods.py` (it had
   only existed to support the state-only design in the first place, so the fix
   was a net simplification, not just a reversal). Also surfaced a mathematical
   fact worth remembering when reasoning about what a grid may safely depend on:
   `"choice"` *always* passes through into the child's own `"lagged_choice"` 1:1
   (enforced by `check_endog_update_function` in `state_choice_space.py`), so a
   grid depending only on `"choice"` can never violate the consistency check,
   structurally, regardless of what else is going on -- unlike an arbitrary
   discrete state, whose pass-through depends entirely on
   `next_period_deterministic_state`. (Caught by a maintainer.)

## Problem

Today, `dcegm` supports "additional continuous states" (e.g. experience) via a
single grid per continuous-state name, declared once in
`model_config["continuous_states"]` and shared identically by every discrete state
in the model (`continuous_states_info["additional_continuous_state_grids"]` in
`pre_processing/check_model_config.py`, meshed once into
`model_structure["continuous_state_space"]` in
`pre_processing/model_structure/model_structure.py`).

We want to let the grid for a continuous state depend on the discrete state (e.g.
different experience grids for different `education` groups). This interacts with
an existing optimization that is load-bearing for performance, not just an
implementation detail:

**Child-state deduplication.** `map_state_choice_to_child_states`
(`pre_processing/model_structure/state_choice_space.py`) maps each
(parent state, choice, stochastic realization) to a child **discrete**-state index —
purely a function of discrete state variables, via
`next_period_deterministic_state` and the stochastic-state grid. Many different
parent state-choices routinely map to the *same* child index; that's the norm, not
an edge case. The batch-creation code
(`pre_processing/batches/algo_batch_size.py`, `single_segment.py`) exploits this by
deduplicating children with `np.unique(child_state_idxs, ...)` and computing the
continuation-value interpolation **once per unique child**, reusing it for every
parent that shares it.

That shared, once-per-child computation
(`egm/interpolate_marginal_utility.py` → `law_of_motion.py:calc_law_of_motion_for_state_choices`)
feeds **the parent's own continuous grid** (as "last period's grid") through a
law-of-motion function keyed by the child's discrete identity, to get the child's
beginning-of-period continuous state/wealth for interpolation. Today this is
harmless because there is only one grid for the whole model, so it doesn't matter
which parent happened to produce a given child. Once grids become state-specific,
it matters: if two parents that share a child disagree on their own grid, the
deduplicated computation can only use one of them, and the other parent's
continuation values would be silently interpolated against the wrong grid.

**The invariant we must enforce:** for every discrete child-state index, all
parent state-choices that map into it must agree on their own continuous grid
(per continuous-state name). This is *not* automatically true once grids can vary
by discrete state — it depends on which discrete variables the grid is a function
of, and whether those variables pass unchanged (or are 1:1) from parent to child.

Example (this repo's model): a grid indexed by `(sex, education, period)` is safe,
because `sex`/`education` are time-invariant endogenous states that
`next_period_deterministic_state` passes through unchanged, and `period` increments
deterministically for everyone — so any two parents sharing a child necessarily
agree on `(sex, education, period)` too, structurally, by construction. A grid
indexed by something like `policy_state`, where the parent→child mapping isn't 1:1
(e.g. a reset rule collapsing several parent values into one child value), could
violate the invariant, and must be caught, not silently mis-solved.

## Design

### Config API

`model_config` holds data/config only — every user-supplied function in `dcegm` is
a separate top-level argument to `create_model_dict` (`state_space_functions`,
`utility_functions`, `budget_constraint`, `stochastic_states_transitions`,
`shock_functions`), never nested inside `model_config`. Grid functions follow the
same convention, as a new optional argument:

```python
create_model_dict(
    ...,
    continuous_grid_functions={
        "experience": grid_func,   # any subset of the model's continuous states
    },
)
```

`grid_func(**discrete_state_kwargs, choice) -> 1D array`, same calling convention
as `sparsity_condition` / `next_period_deterministic_state` (called with the full
state-choice dict as kwargs -- discrete state *and* `"choice"`; the callable picks
out whichever fields it needs, via
`determine_function_arguments_and_partial_model_specs`, same as those two). See
Status item 4 above for why `"choice"` is included: grids live on the
state-choice space, the same granularity the solution itself is stored at, not
the bare state space the feature was originally scoped to. Every state-choice's
returned grid must have the same length as every other state-choice's, for the
same continuous-state name — validated eagerly (see Phase 0), not left to fail
downstream as a shape mismatch deep in `jax.lax.scan`.

A continuous-state name with no entry in this dict keeps exactly today's behavior:
one global grid from `continuous_states_info["additional_continuous_state_grids"]`.
`process_continuous_grid_functions` (in `process_model_functions.py`) validates the
dict, wraps every continuous state's grid into one uniform callable either way, and
also returns the plain list of state-specific names (`model_funcs["state_specific_continuous_grid_names"]`)
so later steps can skip checking names left at their default.

### Why *not* precompute and carry a `(n_states, n_points)` table

An earlier version of this plan proposed eagerly evaluating `grid_func` for every
state during model-structure creation, storing the result as a
`(n_states, n_points)` array in `model_structure`, and gathering rows from it
throughout solving. That does not match how every other user-supplied,
state-dependent function in this codebase is treated:

- `sparsity_condition` and `next_period_deterministic_state` *are* called once per
  state/state-choice during model-structure construction — but only as part of a
  one-time, plain-Python/NumPy enumeration (`pre_processing/model_structure/state_space.py`,
  `state_choice_space.py`), never materialized into a table that then flows through
  `jax.lax.scan`.
- `next_period_continuous_state`, `compute_assets_begin_of_period`, etc. are
  explicitly evaluated **on demand, per batch, inside the jitted solve** —
  `law_of_motion.py` says so directly: "computed on demand per batch/period instead
  of upfront for the whole state space."
- A `(n_states, n_points)` table would also usually be mostly redundant memory: if a
  grid depends on 2-3 discrete variables (e.g. `sex`, `education`), most rows of
  `state_space` are duplicates of each other along those dimensions, multiplied by
  every period/lagged-choice/stochastic-state combination.

So `grid_func` should be processed into `model_funcs` alongside the other
user-supplied functions (`pre_processing/model_functions/process_model_functions.py`),
and evaluated **on demand via `vmap`**, wherever an "own grid" is actually needed
inside a batch computation — exactly the same treatment as
`next_period_continuous_state` already gets. Nothing new is carried around; we add
one more small, cheap, pure function to the set that's already called this way.

The **only** place that needs to look at *every* state's grid is the Phase 1
consistency check below, and that's a one-time NumPy-side pass during model
structure construction (the same phase that already enumerates the full state
space to build `state_space` and `map_state_choice_to_child_states` in the first
place) — not something stored or threaded through the solve.

## Phases

Each phase should land with its own tests and pass the full existing `dcegm` test
suite unmodified (with `continuous_grid_functions` absent) before moving
on — that absence-case regression run is the tripwire against silently changing
behavior for every current `dcegm` user, and should be treated as blocking at every
phase, not just at the end.

### Phase 0 — config plumbing, no behavior change — **done**

- `pre_processing/setup_model.py`: `continuous_grid_functions: Dict[str, Callable] = None`
  is a new top-level argument to `create_model_dict` / `create_model_dict_and_save`
  / `load_model_dict`, **and to `dcegm.setup_model` (`interfaces/model_class.py`)**
  — the actual public entry point users call, which is easy to miss since it wraps
  the three functions above rather than calling them directly. It was missed in
  the first pass; without it the feature was unreachable through the normal API.
- `pre_processing/model_functions/process_model_functions.py`: new
  `process_continuous_grid_functions`, validating keys are a subset of the model's
  continuous states and values are callable, then processing each `grid_func` into
  `model_funcs["continuous_grid_functions"]` the same way
  `next_period_deterministic_state` is processed, keyed by continuous-state name.
  Names without a user callable get a trivial wrapper that returns the existing
  global grid regardless of state (so downstream code always has *a* callable per
  name, never a branch on "is this state-specific"). Also returns the plain list of
  state-specific names as `model_funcs["state_specific_continuous_grid_names"]`.
- The one-time enumeration needed for validation lives in
  `pre_processing/model_structure/continuous_state_grids.py`
  (`evaluate_state_specific_continuous_grids`), called from `model_structure.py`
  right after `create_state_space`. Only evaluates names actually made
  state-specific (skips the constant-wrapper names — trivially consistent).
- Tests: `tests/test_state_specific_continuous_grids.py` — key omitted → constant
  wrapper reproduces the default grid; non-dict / unknown-key / non-callable
  values rejected; inconsistent length across states → clear `ValueError`.

### Phase 1 — the child-sharing consistency check — **done**

- Location: `pre_processing/model_structure/state_choice_space.py`, in
  `create_state_choice_space_and_child_state_mapping`, right after
  `map_state_choice_to_child_states` is fully built, before
  `test_child_state_mapping` — same place, same error style as that existing check.
- Logic: for each continuous-state name with a user-supplied `grid_func`:
  1. Evaluate `grid_func` once per row of `state_space` (plain Python/NumPy loop,
     one-time cost — see Phase 0 note; can share the same evaluation if convenient).
  2. Group state-choice rows by child index, using the *same* grouping mechanism
     the batch code uses later (`np.unique(child_idxs, return_inverse=True)`) —
     deliberately identical, not just equivalent, so this check provably guards the
     exact invariant Phase 4's optimization depends on.
  3. Within each group, assert the grid of `map_state_choice_to_parent_state` is
     identical (`np.array_equal`) across every row in the group.
  4. On failure: raise `ValueError` naming the continuous-state, two example parent
     states that share a child but disagree on their grid, and the child they
     share — matching the formatting already used in `test_child_state_mapping`.
- Tests:
  - Grid depends only on `(sex, education)` → passes across a multi-period,
    multi-choice toy model.
  - Same toy model, grid artificially made to depend on `lagged_choice` in a case
    where two different lagged choices funnel into the same child → fails with
    `pytest.raises(ValueError)`, assert the message names the right states.
  - A model using the `sparsity_condition` proxy mechanism → confirm the check
    operates on `map_state_choice_to_parent_state` (always a valid, non-proxied
    state by construction) and isn't confused by proxied children.
  - Two additional continuous states, only one state-specific → check applies only
    to that one.

### Phase 2 — feed a representative parent's grid into the law of motion, on demand — **done (merged with Phase 4 and Phase 5)**

These three phases turned out not to be separable. Phase 2 alone (evaluate each
state's own grid on demand, but keyed by whichever discrete-state dict happened to
be passed in) is not just incomplete without Phase 4's plumbing — it's actively
wrong, because the caller that's actually reachable in the main solve path
(`interpolate_marginal_utility.py`, called with the *child's* discrete identity)
would silently select the grid using the child instead of the parent. There's no
way to land a correct Phase 2 without also landing the representative-parent
plumbing Phase 4 was for, so they shipped together, and Phase 5 (the last-two-periods
special case, which has exactly the same problem via its own separate code path)
went in at the same time rather than left temporarily broken.

**The core distinction, load-bearing throughout:** `calc_law_of_motion_for_state_choices`
now takes two separate discrete-state dicts, not one — `state_choice_vec` (the
state the law-of-motion *function itself* is evaluated at — the child in the main
solve path, since e.g. its `lagged_choice` genuinely is part of the transition
rule) and `grid_state_dict` (whose *own grid* supplies the input values fed
through that function — must be a representative parent, never the child). See
the docstring of `calc_law_of_motion_for_state_choices` in `law_of_motion.py` for
the full reasoning. When `grid_state_dict` is omitted it falls back to
`state_choice_vec` itself, which is correct for callers with no real parent/child
relationship to trace (the whole-state-space debug entry point
`calc_cont_grids_next_period`) and harmless for any caller that predates this
feature (grids are constant, so it doesn't matter which identity selects them).

What landed:

- `law_of_motion.py`: `compute_own_continuous_grid_combos` evaluates one state's
  grid_func per continuous-state name and meshes them (mirrors the meshgrid
  construction in `model_structure.py`, but on demand, per state, via
  `model_funcs["continuous_grid_functions"]`). `calculate_continuous_state`
  (the original, broadcast/outer-product function — still used directly by the
  whole-state-space debug path and directly unit-tested in
  `tests/test_law_of_motion.py`) was **not** changed; a new
  `calculate_continuous_state_on_own_grids` (paired vmap, `in_axes=(0, 0, ...)`)
  was added alongside it for the on-demand/state-specific path instead, to avoid
  touching an existing tested public contract.
- `pre_processing/batches/child_state_dedup.py` (new): the dedup logic that used to
  be duplicated between `algo_batch_size.py` ("largest_block" mode) and
  `single_segment.py` ("period_max" mode) is now one shared
  `compute_child_dedup_for_batch`, extended to also return, per unique deduplicated
  child state-choice, the state index of a representative parent (found via
  `np.unique(..., return_index=True)` on the child-state dedup — any one parent
  works, since Phase 1 already guarantees every parent sharing a child agrees on
  its grid). Threaded through `correct_for_uneven_last_batch` /
  `prepare_and_align_batch_arrays` as a new batch-info field,
  `representative_parent_state_idx`, padded the same way as
  `child_state_choice_idxs_to_interp`.
- `backward_induction.py` / `solve_single_period.py`: the new field rides along in
  `xs` through `jax.lax.scan`; `solve_single_period` gathers
  `model_structure["state_space_dict"]` at that index to build `grid_state_dict`
  and passes it to `interpolate_value_and_marg_util`.
- `interfaces/inspect_solution.py`: a debug entry point that manually replays one
  batch of `solve_single_period` outside of `lax.scan` needed the same new `xs`
  element and `state_space_dict` argument.
- `pre_processing/batches/last_two_periods.py` / `final_periods.py`: the two-period
  special case has its own, undeduplicated version of the same problem —
  `solve_final_period` computes the law of motion for *every* final-period
  state-choice individually (no batching/dedup at this stage), so
  `add_last_two_period_information` computes a representative
  second-to-last-period parent state for every final-period state-choice (falling
  back to self-reference for any final-period state not actually reachable from
  the second-to-last period — an existing, already-flagged-elsewhere edge case,
  see `test_child_state_mapping`'s warning).

Tests:

- `tests/test_law_of_motion_state_specific_grid.py`: unit tests of
  `compute_own_continuous_grid_combos` / `calculate_continuous_state_on_own_grids`
  in isolation, including a synthetic case with `state_space_dict` (child) and
  `grid_state_dict` (parent) deliberately holding *different* discrete-state
  values, asserting the grid used comes from `grid_state_dict` — this is the
  direct regression test for the bug described above.
- `tests/test_state_specific_grid_end_to_end.py`: the full pipeline, on the real
  5-period `with_cont_exp` toy model (exercises both the main backward-induction
  loop and the last-two-periods special case in one solve). A grid_func that
  returns the model's existing default grid reproduces the unmodified baseline
  solve **bit-for-bit** — the critical regression check that Phase 2/4/5 are a
  no-op whenever grids aren't actually state-specific. A genuinely period-varying
  grid solves without crashing/NaN-ing (beyond the pre-existing, unrelated
  variable-endogenous-grid NaN padding, confirmed identical in both solves) and
  produces different numbers than the baseline — but see the note below on why
  this alone isn't a correctness proof, and why Phase 3 added a stronger test.
- Full `dcegm` suite passes throughout (992 tests on the branch as of this
  writing), including direct callers of the functions touched here
  (`tests/test_law_of_motion.py`, `tests/test_partial_and_interfaces.py`) that
  predate this feature and don't pass the new optional arguments at all.

### Phase 3 — a state's own grid for its own solve, not just for its parents' transitions — **done (both FUES and Druedahl-Jorgensen paths)**

Originally scoped as "child's own storage-grid, on demand in
`interpolate_marginal_utility.py`" — evaluating each deduplicated child's
`grid_func` for the interpolation axis instead of reading the shared
`additional_continuous_state_grids`/`continuous_state_space`. That turned out to
be only one of three places reading the shared/global grid where they should read
a state's own — writing the Phase 2/4/5 end-to-end test surfaced two more:

1. **Interpolating a child's stored solution** — `egm/interpolate_marginal_utility.py`'s
   `_interpolate_value_and_marg_util_2d_irregular` (**done**) /
   `_interpolate_value_and_marg_util_nd_regular` (**not done**, see below) took
   `additional_continuous_state_grids` / `continuous_state_space` (global) as the
   interpolation axis for the child's stored policy/value. No invariant needed
   here — a child's own grid is always well-defined per child, unlike the
   parent-selection problem Phase 1/2 solve — just a correct per-(deduplicated)-
   child `grid_func` evaluation instead of a shared array. `state_choice_vec`
   (already each child's own identity in this function) is what's used, since
   we're reading each child's *own* stored solution here, not transitioning into
   it.
2. **A state's own EGM candidate generation** (**done**) — `egm/solve_euler_equation.py`'s
   `calculate_candidate_solutions_from_euler_equation`, which determines what a
   state's own storage combo-axis actually *represents* when that state is itself
   being solved (not interpolated as someone else's child), also took
   `continuous_state_space` (global) unconditionally. This was the more
   fundamental gap: before this, no state had actually solved or stored its own
   problem on its own grid anywhere in the pipeline — Phase 2/4/5 alone only
   changed which grid values are fed *into* a parent→child transition, not what a
   state's own stored solution is indexed against. `state_choice_mat` (this
   period's own state-choices, no dedup at this stage) is each row's own
   identity, so — like item 1 — no representative-parent selection is needed here,
   unlike `law_of_motion.py`'s grid selection for a transition *into* a state.
3. **The final period's own value computation** (**done**, found via the strong
   test below) — `final_periods.py`'s `solve_final_period` has its own separate
   code path for computing the terminal period's value (no continuation value
   needed, so it doesn't go through `solve_euler_equation.py` at all) via
   `calc_value_and_budget_for_each_gridpoint`, and that also still read
   `model_structure["continuous_state_space"]` (global) for the final period's
   own combo axis. This is the final-period analog of item 2, on a separate code
   path with nothing to inherit the fix from — easy to miss, and initially
   missed; see status item 3 above.

**Druedahl-Jorgensen (ND regular) path — done.** `interpolation/interpnd_regular.py`'s
`_precompute_regular_indices_and_weights` computed index/weight pairs via
`get_index_high_and_low(grid_1d, points)` where `grid_1d` was *one* 1D array shared
across every child for a given continuous-state name. Fixed by adding a parallel
`interpnd_policy_and_value_for_child_states_on_own_regular_grids` (plus
`_precompute_interp_objects_own_grids`/`_precompute_regular_indices_and_weights_own_grids`)
that vmaps `get_index_high_and_low` itself over children, pairing each child's own
grid (shape `(n_children, n_points)`) with its own query points via
`jnp.take_along_axis`, instead of relying on one shared 1D grid broadcasting
against many points. Deliberately added *alongside* the original shared-grid
functions rather than changing them — those are directly, thoroughly validated
against scipy's `RegularGridInterpolator` in `tests/test_interpnd_regular.py`, and
that contract is still used by other callers, so it stays untouched. Wired into the
solve path via `egm/interpolate_marginal_utility.py`'s
`_interpolate_value_and_marg_util_nd_regular`, which now builds each child's own
grids with `vmap(compute_own_continuous_grids_raw, in_axes=(0, None, None))`
(the same on-demand, per-child pattern as the rest of this feature) instead of
reading `continuous_state_space`/`additional_continuous_state_grids`.
`compute_own_continuous_grids_raw` was added to `law_of_motion.py` alongside the
existing `compute_own_continuous_grid_combos` (unmeshed vs. meshed — the ND-regular
interpolation needs each dimension's own grid separately, not a dense meshgrid).

Validated at two levels: `tests/test_interpnd_regular_own_grids.py` checks the raw
interpolation math (per-child grids against scipy ground truth, and a regression
check that identical-per-child grids reproduce the original shared-grid function
bit-for-bit), and `tests/test_state_specific_grid_dj_end_to_end.py` runs the same
"constant but different grid matches direct `model_config` declaration" strong
check as the FUES test below, through a real multi-period solve — built by taking
the existing `with_cont_exp` toy model and adding `assets_begin_of_period` to
`continuous_states` plus `upper_envelope={"method": "druedahl_jorgensen"}` (no new
toy model needed; `check_model_config.py` only forces DJ when there's more than one
*additional* continuous state, but doesn't forbid using DJ with just one).

Tests (`tests/test_state_specific_grid_end_to_end.py`,
`tests/test_law_of_motion_state_specific_grid.py`):

- `test_constant_but_different_grid_matches_direct_model_config_declaration` — the
  strong correctness check that caught bug 3 above. A *constant* grid delivered
  via `continuous_grid_functions`, but with different values than the model's own
  default, must reproduce a model where that same grid is declared directly in
  `model_config["continuous_states"]` (today's unmodified mechanism) bit-for-bit.
  This is a materially stronger check than "reproduces the *default* grid" (passes
  trivially if the on-demand path silently falls back to the default) or
  "produces *different* numbers than the default" (passes even if the numbers are
  wrong, as long as they're not exactly the default) — it can only pass if the
  on-demand grid is genuinely the one being used, correctly, everywhere it needs
  to be. Runs against the real `with_cont_exp` toy model, FUES/2D path.
- The two weaker Phase 2/4/5 end-to-end tests above are still kept (they still
  cover the batching/dedup correctness Phase 2/4/5 is about specifically), now
  documented as weaker deliberately so a future reader doesn't over-trust them.

### Phase 6 — readers: simulation & interfaces — **done**

Three separate reader code paths all still read the shared/global grid when
mapping a query's continuous value onto the stored solution — each is a distinct
place the Phase 2-5 solve-side fix doesn't reach, since none of them go through
`solve_single_period.py`:

1. **`simulate_all_periods`** (`simulation/simulate.py` →
   `interpolation/simulation_interp.py`'s `interpolate_policy_and_value_for_all_agents`)
   — the vmapped per-agent FUES (`interp2d_policy_and_value_function`) and DJ
   (`interpnd_policy_and_value_function`) inner functions took a precomputed shared
   `continuous_state_space`/`additional_continuous_state_grids` as a constant vmap
   argument. Fixed by passing `continuous_grid_functions` through as the constant
   argument instead (callables-as-vmap-constants was already an established
   pattern here — `compute_utility` is passed the same way), and computing each
   agent-choice's own grid inside the inner function via
   `compute_own_continuous_grids_raw`/`compute_own_continuous_grid_combos`, using
   `state_choice_vec = {**state, "choice": choice}` as the (self-referential)
   representative — the state-choice being queried is always its own
   representative for reading back its own stored solution, unlike the
   parent→child transition case Phase 1/2 solve. Threaded from
   `simulate_all_periods` using the *original* `model_funcs["continuous_grid_functions"]`
   (matching `compute_utility`), not `alt_model_funcs_sim` — the stored solution
   was solved with the original model's grids, so a counterfactual-behavior
   simulation (different `alt_model_funcs_sim`) must still read it back with the
   grids it was actually solved on.
2. **`choice_values_for_states`/`choice_policies_for_states`**
   (`interfaces/interface.py` → `interpolation/interp_interfaces.py`) — same shared
   `model_structure["continuous_state_space"]` read, same self-referential fix
   (`state_choice_vec` is already available in each of the three
   `interpolate_*_for_state_and_choice` functions and their shared DJ helper
   `_interp_policy_and_value_multidim_dj_for_state_choice`).
3. **`interfaces/inspect_solution.py`** — turned out *not* to need a fix: its use of
   `model_structure["continuous_state_space"]` is only for reading output-array
   shape (`n_continuous_state_combinations`, same combo *count* regardless of grid
   values) and it otherwise mirrors `solve_single_period.py` exactly (already
   fixed in Phase 2/4/5), since it's the debug/partial-solve entry point, not an
   independent reader.

Test: extended `tests/test_state_specific_grid_end_to_end.py` and
`tests/test_state_specific_grid_dj_end_to_end.py` with the same strong "constant
but different grid matches direct `model_config` declaration" bit-for-bit check as
Phase 3, applied through `.simulate()` and through `choice_values_for_states`/
`choice_policies_for_states` — chosen deliberately over a weaker
"doesn't crash"/"differs from default" check, per the lesson from bug 3 in the
Status section above (weaker end-to-end tests passed while a real bug was still
present in `final_periods.py`).

### Phase 7 — end-to-end validation — **done**

- Ground truth (`tests/test_state_specific_grid_ground_truth.py`): rather than
  building a new toy model from scratch, reuses `with_cont_exp` unchanged and adds
  a `group` dimension via `model_config["deterministic_states"]` — a pre-existing
  `dcegm` mechanism (`deterministic_states.py`) whose default law of motion already
  leaves it unchanged period to period, so it structurally satisfies Phase 1's
  consistency check without needing a custom `next_period_deterministic_state`.
  `group` doesn't enter `with_cont_exp`'s utility, budget, or transition functions
  at all, so group 0 and group 1's economics are identical except for which grid
  the solution is stored/interpolated on. Solved two ways: once as one combined
  model (`group` ∈ {0, 1}, grid depends on `group` via `continuous_grid_functions`)
  and once as two fully separate single-group models, each declaring its group's
  grid directly via `model_config["continuous_states"]` (today's pre-existing,
  unmodified mechanism). Compared via `choice_values_for_states`/
  `choice_policies_for_states` (the same public query interface validated in Phase
  6) — the combined model's group-0 and group-1 answers each match their
  respective separate single-group model bit-for-bit.
- Negative test: already covered, not written new. `test_grid_depending_on_lagged_choice_fails`
  and `test_grid_on_state_that_does_not_pass_through_1to1_fails` in
  `tests/test_state_specific_continuous_grids.py` (built during Phase 1) use this
  same `group` mechanism to construct exactly this case — a grid depending on a
  variable that does not survive the parent→child transition 1:1 — and confirm
  `create_state_choice_space_and_child_state_mapping` raises at model-structure
  build time, before any solve happens.

## Open questions

1. ~~Should `assets_end_of_period` / `dj_wealth_grid` ever become state-specific
   too?~~ **Answered: yes**, `assets_begin_of_period` explicitly, and
   `assets_end_of_period` was included alongside it for consistency.
   `process_continuous_grid_functions` already accepts both as valid keys (Phase
   0). What's still missing is Phase 3 item 2 above — `solve_euler_equation.py`'s
   EGM candidate generation, which actually determines what a state's own wealth
   grid *is*, doesn't consume `continuous_grid_functions` for these two names yet.
   So the config-level door is open but nothing behind it is wired up for the
   wealth grids specifically.
2. Land phases incrementally on `main` in the `dcegm` submodule's own repo
   (`OpenSourceEconomics/dcegm`), or as one long-lived feature branch merged at the
   end? Still open. Given how much Phase 2 turned out to depend on Phase 4 (and
   Phase 3 is turning out similarly entangled with `solve_euler_equation.py`),
   "incremental" may end up meaning "incremental at the level of these merged
   phases," not literally one phase per PR.
