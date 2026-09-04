# State-choice-specific `assets_end_of_period` / `assets_begin_of_period` grids

> **Note.** This page is AI-drafted (with Claude, based on reading the source and a
> design discussion with a maintainer) and is a *plan*, updated as phases land and
> design decisions change. It replaces an earlier plan doc at this same path for the
> now-complete "state-specific continuous state grids" feature (state-choice-specific
> grids for *additional* continuous states like "experience"), which shipped in full —
> see git history for that design if needed. This doc covers the follow-on work: making
> the two continuous states outside that feature's scope, `assets_end_of_period` and
> `assets_begin_of_period`, state-choice-specific too, plus a config ergonomics change
> (the `None`-grid convention) that applies to all state-choice-specific continuous
> states, not just these two.

## Status

**Done, tested, passing the full `dcegm` suite.** All phases below landed, with two
corrections to the original design (see "What changed from the original plan"
below): `assets_end_of_period` does *not* get the `None`-grid convention after all,
and a genuine pre-existing bug (unrelated to this work) had to be fixed as a
prerequisite for testing `assets_begin_of_period` state-specificity at all.

### What changed from the original plan

1. **`assets_end_of_period` does not support the `None`-grid convention, and is the one
   name exempt from the (later-added) requirement that a declared array be `None`
   whenever a `continuous_grid_functions` entry exists.** `check_model_config.py` uses
   `len(model_config["continuous_states"]["assets_end_of_period"])` to compute
   tuning-parameter defaults (`n_constrained_points_to_add`, `extra_wealth_grid_factor`
   -derived `n_total_wealth_grid`) — this runs unconditionally, for every
   `upper_envelope` method, before `continuous_grid_functions` is even known and before
   any state-choice exists to pin a deferred size against; it isn't "for FUES"
   specifically, even though only the `fues` branch actually ends up consuming the
   result (`druedahl_jorgensen` overwrites it with its own `assets_begin_of_period`
   -derived formula). Making its size deferrable would mean reordering
   `check_model_config.py` so this `len()` call isn't unconditional, a materially
   bigger and riskier change than this work needed. `assets_end_of_period` still fully
   supports state-choice-specific *values* via `continuous_grid_functions` — a real
   array must always be declared regardless, and is simply ignored once a grid_func is
   given, the one case where that's allowed rather than rejected.
2. **A genuine, pre-existing bug turned up as a blocker**: `final_periods.py`'s
   `solve_final_period` had a `has_additional_continuous_state` branch that, for
   models *without* an additional continuous state, unconditionally read
   `wealth_child_states_final_period` (the law-of-motion's `assets_end_of_period`-based
   output) as the final-period storage grid — correct for FUES, but silently wrong
   for Druedahl-Jorgensen (which should use `assets_begin_of_period` there instead).
   This was apparently never exercised before: every existing DJ test in this repo
   used a toy model with an additional continuous state, which goes through a
   different (correct) branch. It surfaced immediately as a shape-mismatch crash
   when writing the first `assets_begin_of_period`-state-specific end-to-end test
   against a no-additional-continuous-state DJ model, and had to be fixed (a new
   `_calc_dj_final_value_no_additional_state` branch) before that work could be
   validated at all. Unrelated to state-specificity itself — the bug is present for
   *any* DJ model without an additional continuous state, state-specific or not.
3. **`assets_begin_of_period` state-specificity together with an additional
   continuous state is now supported, but only when `skip_endog_grid_storage` is
   `True`** (`upper_envelope["method"] == "druedahl_jorgensen"` and at least two
   choices), enforced by `process_continuous_grid_functions`. The n-D regular
   (Druedahl-Jorgensen) interpolation path
   (`interpnd_policy_and_value_for_child_states_on_own_regular_grids`) now threads
   `wealth_grid` per child too (shape `(n_child_state_choices, n_wealth)`),
   mirroring `additional_continuous_state_grids_per_child`; the matching per-child
   computation (`vmap(compute_own_dj_wealth_grid, ...)`, the child's own identity,
   same reasoning as the simple-1D-DJ branch) lives in
   `_interpolate_value_and_marg_util_nd_regular`
   (`interpolate_marginal_utility.py`). With a single choice the Druedahl-Jorgensen
   upper envelope is skipped entirely (see `check_model_config.py`), so the stored
   endogenous grid there is a genuinely different, non-fixed grid — a
   state-specific `assets_begin_of_period` has nothing to plug into in that case,
   and remains rejected.

## Problem

`continuous_grid_functions` (the top-level `create_model_dict` argument from the prior
feature) is already *accepted* for `"assets_end_of_period"` and `"assets_begin_of_period"`
at the config-validation layer: `process_continuous_grid_functions`'s `default_grids`
dict includes both, a user-supplied grid_func for either gets wrapped correctly, and
`state_specific_continuous_grid_names` (just `list(continuous_grid_functions.keys())`,
not filtered) would even run either through the Phase-1 child-sharing consistency check.

But nothing downstream ever calls `continuous_grid_functions["assets_end_of_period"]` or
`["assets_begin_of_period"]`. Every consumer reads a single, model-wide global array
directly instead:

- `assets_end_of_period`: always `continuous_states_info["assets_grid_end_of_period"]`,
  read directly in `law_of_motion.py`, `solve_euler_equation.py`,
  `interpolate_marginal_utility.py`, `final_periods.py`.
- `assets_begin_of_period` (Druedahl-Jorgensen's fixed common wealth grid, "m_grid"):
  always `continuous_states_info["assets_begin_of_period"]` / its derived
  `dj_wealth_grid`, read directly in `upper_evelope_wrapper.py`'s `drued_jorg_jax(...)`
  call and broadcast everywhere via `broadcast_dj_wealth_grid`
  (`sol_container.py`, `interface.py`, `simulation_interp.py`) whenever
  `skip_endog_grid_storage` is `True`.

So today a user can pass a `continuous_grid_functions` entry for either name, it
validates cleanly, and is then silently ignored at solve time. That's the gap.

**Deliberately out of scope / unaffected:** the *additional* continuous states
mechanism (`additional_continuous_state_names`, `compute_own_continuous_grid_combos`,
the representative-parent threading through batches) stays exactly as it is,
scoped to continuous states other than these two — `check_model_config.py` already
excludes both names from `additional_continuous_states`, and that boundary is not
changing. `assets_end_of_period`/`assets_begin_of_period` get their own, separate
wiring, reusing the same underlying ideas (on-demand evaluation, representative
parent where needed) rather than being folded into the existing list.

## Design

### Two different mechanisms, not one

The two names need different treatment because they play structurally different
roles in the solve:

**`assets_end_of_period`** is an *input to the law of motion*: it's the exogenous
savings grid a parent's transition is evaluated over
(`calc_beginning_of_period_assets_for_single_state`'s `asset_end_of_previous_period`
comes from iterating over this grid, feeding the child's own
`compute_assets_begin_of_period` budget equation). This is exactly the same
parent→child relationship that made the *additional continuous states* feature need
representative-parent threading in the first place: the child's own identity is
correct for the transition call itself, but the grid *values* fed into it must come
from a representative parent's own grid. So `assets_end_of_period` needs the same
representative-parent mechanism, applied to this one additional name.

**`assets_begin_of_period`** is not a law-of-motion input at all — it's the *output*
common grid that Druedahl-Jorgensen's upper envelope interpolates onto and stores
against (`m_grid`). Reading a child's own stored solution back doesn't need a
representative parent (same reasoning as reading a child's own *additional*
continuous-state combo grid back during interpolation — see
`interpolate_marginal_utility.py`'s "own grid" pattern) — it just needs to know the
right grid was used at solve time for *this* state-choice, indexed by its own
identity, no parent involved. And it only matters at all when
`skip_endog_grid_storage` is `True` (Druedahl-Jorgensen with ≥2 choices) — that's
precisely the case where the real per-state-choice `endog_grid` isn't stored, because
every state-choice's "endogenous" grid is by construction this fixed array. When
`skip_endog_grid_storage` is `False` (Druedahl-Jorgensen with 1 choice, or FUES), the
real stored `endog_grid` is read/used normally and `assets_begin_of_period` plays no
role in interpolation — so state-specificity for this name is meaningless outside
the `skip_endog_grid_storage=True` case, and should be gated on it explicitly (e.g.
rejected or ignored otherwise, mirroring the existing FUES/`assets_begin_of_period`
mutual-exclusion check in `check_model_config.py`).

### The `None`-grid convention

Separately, for *any* state-choice-specific continuous state (this includes the
existing "additional continuous states" mechanism too, retroactively, plus these two
new names): today, once a `continuous_grid_functions[name]` entry is given, the
*values* declared in `model_config["continuous_states"][name]` are entirely unused —
but the array must still be supplied, purely so its `len()` can be read off to fix
the grid size and validate every state-choice's own grid against it. That's a
redundant, awkward requirement: fabricating a same-length dummy array just to declare
a size.

New convention, now enforced in both directions: `model_config["continuous_states"][name]
= None` is required whenever a matching `continuous_grid_functions[name]` is given, and
required to be non-`None` otherwise. `None` with no function raises (nothing to build the
grid from); a real array *with* a function also raises (the array would be silently
unused, which is exactly the awkward, easy-to-misread state this convention exists to
rule out — see `process_continuous_grid_functions`). The key must still be *present* in
`model_config["continuous_states"]` (so a genuinely missing continuous state is still
caught as a missing key, not confused with this explicit opt-in). When `None` is
declared, the grid's size is pinned by evaluating the grid_func once on a representative
state-choice row instead of reading a precomputed length. `assets_end_of_period` is the
one name exempt from this: it must always keep a real array, function or not (see below).

**Architectural consequence:** this moves *when* the expected length becomes known.
Today it's read from `model_config["continuous_states"][name]` in
`check_model_config.py`, before the state-choice space exists. Pinning it by
evaluation requires an actual state-choice row to call the function on, which only
exists after `create_state_choice_space_and_child_state_mapping` builds
`state_choice_space` — so "determine expected length for this name" needs to move to
alongside `evaluate_state_specific_continuous_grids` in `continuous_state_grids.py`
(which already evaluates every row's grid_func output for the consistency check;
deriving `expected_length` from row 0's own output shape there is a natural
extension, not a bolt-on). `n_continuous_state_combinations` (which sizes the global
solution containers, currently computed early in `check_model_config.py`) becomes a
two-stage value: known immediately for names with a real declared array, resolved
only after state-structure construction for names declared as `None`.

## Phases

1. **Config layer — done.** `None` allowed (and, once a `continuous_grid_functions[name]`
   entry exists, *required*) in `model_config["continuous_states"][name]` for the
   additional continuous states and `assets_begin_of_period`; `assets_end_of_period` is
   exempt in both directions (see above) and must always keep a real array.
   `process_continuous_grid_functions` raises if a `None`-eligible name is declared
   `None` with no matching function, and equally raises if it's declared as a real array
   *with* one. Also added: a validation rejecting `assets_begin_of_period`
   state-specificity together with any additional continuous state.
2. **Grid-size pinning — done.** `evaluate_state_specific_continuous_grids` (in
   `continuous_state_grids.py`) pins the length of a `None`-declared grid from the
   first state-choice row it evaluates, alongside its existing per-row consistency
   check. `_merge_resolved_state_specific_lengths` (in `state_choice_space.py`)
   folds the resolved length back into `n_continuous_state_combinations` and, for
   `assets_begin_of_period`, `n_total_wealth_grid`.
3. **`assets_end_of_period` representative-parent threading — done.**
   `calc_law_of_motion_for_state_choices` evaluates `assets_end_of_period` from
   `representative_parent_state_choice_vec` (the same representative-parent dict the
   additional continuous states already use), replacing the old shared
   `assets_grid_end_of_period` array parameter. Its "own grid" role (this
   state-choice solving its own EGM problem in `solve_euler_equation.py`, or its own
   terminal value in `final_periods.py`'s FUES branch) is self-referential, evaluated
   from `state_choice_vec` directly, no representative parent needed there.
4. **`assets_begin_of_period` own-grid threading — done.** New
   `compute_own_dj_wealth_grid` helper in `law_of_motion.py`, used wherever
   `dj_wealth_grid`/`broadcast_dj_wealth_grid` used to be read as a shared array:
   `upper_evelope_wrapper.py`'s DJ branch (solve), `interpolate_marginal_utility.py`'s
   simple-1D-DJ interpolation branch (solve, reading a child's own stored solution),
   and `interp_interfaces.py`'s three `interpolate_*_for_state_and_choice` functions
   plus `sol_interface.py`'s raw-grid accessor (readers). `broadcast_dj_wealth_grid`
   itself now returns a zero placeholder when `dj_wealth_grid` is `None` — safe
   because every remaining caller of it is on the n-D-regular (shared-grid-only)
   path, never actually read when state-specific.
5. **Readers — done**, folded into phase 4 above (the two mechanisms turned out to
   share almost all their reader-side call sites, so weren't worth separating).
   `simulate()`'s `interpolate_policy_and_value_for_all_agents` computes its own
   per-agent Druedahl-Jorgensen wealth grid the same way.
6. **End-to-end validation — done.** Strong bit-for-bit "constant-but-different-grid
   matches direct declaration" tests for both names, through the solve path and all
   reader paths (`simulate()`, `choice_values_for_states`/`choice_policies_for_states`):
   `tests/test_assets_end_of_period_state_specific.py`,
   `tests/test_assets_begin_of_period_state_specific.py`. The latter also confirmed
   the pre-existing `final_periods.py` bug fix and the multidim-rejection validation.

## Open questions (resolved)

- Which state-choice row is "representative" for pinning a `None`-declared grid's
  size: row 0 of `state_choice_space`, unconditionally. Every other row's grid_func
  output is validated against that pinned length in the same pass (mismatches raise,
  same as the pre-existing declared-length validation) — a period-dependent grid_func
  that changes size would already be caught this way, no separate check needed.
- The representative-parent machinery for `assets_end_of_period` reuses
  `representative_parent_state_choice_vec` directly (no separate threading): it's evaluated
  alongside the additional continuous states, from the same representative-parent
  dict, inside `calc_law_of_motion_for_state_choices`.
