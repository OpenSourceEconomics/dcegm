# State-specific continuous state grids — implementation plan

> **Note.** This page is AI-drafted (with Claude, based on reading the source and a
> design discussion with a maintainer) and is a *plan*, not a description of shipped
> behavior. Nothing described here exists in the code yet. It is meant as a working
> reference for whoever implements this feature (human or AI), to be updated as
> phases land and design decisions change. See also
> [`batching.rst`](batching.rst), which documents the mechanism this feature
> modifies (deduplication of child states across parent state-choices).

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

`grid_func(**discrete_state_kwargs) -> 1D array`, same calling convention as
`sparsity_condition` / `next_period_deterministic_state` (called with the full
discrete-state dict as kwargs; the callable picks out whichever fields it needs,
via `determine_function_arguments_and_partial_model_specs`, same as those two).
Every state's returned grid must have the same length as every other state's, for
the same continuous-state name — validated eagerly (see Phase 0), not left to fail
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

### Phase 0 — config plumbing, no behavior change

- `pre_processing/setup_model.py`: add `continuous_grid_functions: Dict[str, Callable] = None`
  as a new top-level argument to `create_model_dict` / `create_model_dict_and_save`
  / `load_model_dict`, following the same convention as `shock_functions` — never
  nested inside `model_config`, which holds data/config only.
- `pre_processing/model_functions/process_model_functions.py`: new
  `process_continuous_grid_functions`, validating keys are a subset of the model's
  continuous states and values are callable, then processing each `grid_func` into
  `model_funcs["continuous_grid_functions"]` the same way
  `next_period_deterministic_state` is processed, keyed by continuous-state name.
  Names without a user callable get a trivial wrapper that returns the existing
  global grid regardless of state (so downstream code always has *a* callable per
  name, never a branch on "is this state-specific"). Also returns the plain list of
  state-specific names as `model_funcs["state_specific_continuous_grid_names"]`.
- **This phase also does the one-time enumeration needed for validation**: while
  building `state_space` (or immediately after), call each `grid_func` once per
  state, check the returned length matches across all states for that name. This
  reuses the same enumeration the state-space construction already does — no new
  full-state-space pass, just an extra check attached to the existing one.
- Tests (new `tests/test_state_specific_continuous_grids.py`):
  - key omitted → `model_structure`/`model_funcs` identical to today.
  - key given, consistent lengths → parses without error.
  - key given, inconsistent length across states → clear `ValueError` naming the
    offending states and continuous-state name.

### Phase 1 — the child-sharing consistency check

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

### Phase 2 — feed the parent's own grid into the law of motion, on demand

- `law_of_motion.py`: `calc_law_of_motion_for_state_choices` currently takes a
  single shared `continuous_state_space` dict. Change it to call each name's
  `grid_func` (from `model_funcs`, Phase 0) via `vmap` over the batch's own
  discrete state, instead of broadcasting one shared array — i.e. replace the outer
  product in `calculate_continuous_state` (`in_axes=(0, None, ...)`, all states
  against one grid) with a paired call where each state's grid comes from its own
  `grid_func(**state)` evaluation (`in_axes=(0, ...)` over states, `grid_func`
  itself un-batched and called fresh per state via the same `vmap`).
- Critical regression test: for a name with no user callable (or a user callable
  that returns the same grid regardless of state), output must be **bit-identical**
  to today's global-grid path. Run against the existing continuous-experience toy
  models (`toy_models/cons_ret_model_with_cont_exp`,
  `toy_models/cons_ret_model_with_exp`) with zero config changes beyond what
  Phase 0 requires.
- New unit test: tiny synthetic 2-3-state example with a genuinely varying grid,
  hand-computed expected transitions.

### Phase 3 — child's own storage-grid, on demand

- `egm/interpolate_marginal_utility.py`: `_interpolate_value_and_marg_util_nd_regular`
  / `_interpolate_value_and_marg_util_2d_irregular` currently take
  `additional_continuous_state_grids` (global, per-name 1D arrays) as interpolation
  axes for the child's stored solution. Switch to evaluating each deduplicated
  child's own `grid_func` on demand (same treatment as Phase 2, mirrored for the
  child side). This side needs no invariant — a child's own grid is always
  well-defined per child — just a correct per-child evaluation instead of a shared
  array.
- Tests: hand-built child value functions on two different grids; confirm each
  child interpolates on its own axis, not the other child's.

### Phase 4 — batch creation: use the (now-verified) shared parent grid

- No change to the dedup *mechanism* — Phase 1 guarantees it's safe to compute
  once per unique child. Only change: `algo_batch_size.py` / `single_segment.py`
  already get a first-occurrence parent index for free from
  `np.unique(..., return_index=True)`; thread that representative parent's state
  through to `solve_single_period.py` so Phase 2's on-demand `grid_func` call uses
  the right (guaranteed-representative) discrete state.
- Test: build a batch by hand, assert the representative-parent grid evaluation
  matches manually looping over every parent in the group and confirming they
  agree (redundant with Phase 1 in production, but a useful direct check here).

### Phase 5 — `final_periods.py`

- The last two periods bypass batching entirely (`solve_last_two_periods`) and need
  the Phase 2/3 treatment applied directly, since there's no dedup step to inherit
  it from.
- Test: extend `tests/test_two_period_continuous_experience.py` with a
  state-specific-grid variant.

### Phase 6 — readers: simulation & interfaces

- `simulation/simulate.py`, `simulation/sim_utils.py`,
  `interpolation/interp_interfaces.py`, `interfaces/model_class.py`,
  `interfaces/interface.py`, `interfaces/inspect_solution.py` — anywhere a
  simulated or queried continuous value is mapped onto the grid to read the stored
  solution must resolve the *current discrete state's* grid first, via the same
  on-demand `grid_func` call, not a shared array.
- Test: extend `tests/test_simulate_continuous_state.py` — simulate a population
  split across states with different grids, confirm each agent reads its own
  state's stored solution.

### Phase 7 — end-to-end validation

- New toy model under `toy_models/` with a grid depending on
  `(sex, education, period)` — structurally guaranteed to satisfy Phase 1.
- Ground truth: solve the same economics two ways — once via the new
  state-specific-grid config, once via fully separate discrete "type" states each
  internally using a plain global grid (today's code, unmodified) — and compare
  policy/value functions state-by-state. They should be identical; it's the same
  economic model expressed two ways.
- Negative test: a config where the grid depends on a variable that does not
  survive the parent→child transition uniformly (the `policy_state` example above)
  → confirm Phase 1 rejects it at model-structure build time, before any solve
  happens.

## Open questions

1. Should `assets_end_of_period` / `dj_wealth_grid` (the savings / Druedahl-Jorgensen
   wealth grid) ever become state-specific too, or is scope strictly the
   "additional continuous states" (experience-like)? This plan scopes to the
   latter only.
2. Land phases incrementally on `main` in the `dcegm` submodule's own repo
   (`OpenSourceEconomics/dcegm`), or as one long-lived feature branch merged at the
   end? Given this is a shared OSS package with other users, incremental landing
   behind the config-key feature flag (absent by default) seems lower-risk.
