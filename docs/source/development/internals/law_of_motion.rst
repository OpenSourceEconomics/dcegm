.. _law_of_motion_internals:

The Law of Motion: Evaluation Granularity
==========================================

.. warning::

   This page is AI-drafted (with Claude, based on reading the source and a design
   discussion with a maintainer). It documents current behaviour as of writing;
   verify against the source if something looks off. The performance claims below
   are reasoned, not measured -- see "What is not measured".

What "the law of motion" means here
------------------------------------

Two user-supplied functions describe how a state transitions into the next period:

- ``budget_constraint`` -- processed into
  ``model_funcs["compute_assets_begin_of_period"]``. Maps the previous period's
  end-of-period assets, this period's income shock, and the discrete state to
  beginning-of-period wealth. This is the *only* law of motion that may declare
  ``choice``, and it means the choice made in the period whose wealth is being
  computed -- see :ref:`choice_dependent_budget` below.

  Its two non-state arguments are deliberately asymmetric in timing, and the names
  say so: ``asset_end_of_previous_period`` really is carried over from ``t-1``,
  while ``income_shock`` is realized in ``t``, the period whose wealth is being
  computed. (It was called ``income_shock_previous_period`` until the shipped
  example models' habit of earning income against ``lagged_choice`` was mistaken
  for a framework property; ``_check_budget_does_not_declare_renamed_income_shock``
  in ``process_model_functions.py`` rejects the old name rather than letting the
  signature filter raise a bare ``KeyError``.)
- ``next_period_continuous_state`` -- optional, only when the model has additional
  continuous states (e.g. continuous experience). Maps this period's continuous
  state to next period's. It may **not** declare ``choice``; doing so raises at
  model build (``_check_choice_not_declared`` in ``process_model_functions.py``).
  The previous period's choice, which is what accumulation processes actually
  need, is the ordinary state variable ``lagged_choice``.

Both are evaluated in
:func:`dcegm.law_of_motion.calc_law_of_motion_for_state_choices`, called from
``egm/interpolate_marginal_utility.py`` (the main solve path, once per batch) and
``final_periods.py`` (the terminal period).

The nesting
------------

``calc_law_of_motion_for_state_choices`` evaluates the budget equation over a
four-dimensional product, one ``vmap`` level per axis:

.. code-block:: text

    for each row                     (state-choice, or child state -- see below)
      for each continuous-state combo point
        for each end-of-period assets grid point
          for each income shock draw
            compute_assets_begin_of_period(...)

Each level strips one axis, so the innermost call sees a single scalar tuple and
the user's function never has to know it is being batched. The grid that supplies
the assets axis is evaluated inside the outermost level, right next to where it
is consumed, rather than precomputed for the whole batch and passed in -- see
``_transitions_for_one_state``.

Parent vs. child: two different state-choices
-----------------------------------------------

This trips people up, so it is worth stating precisely. Two distinct state-choice
dicts flow into this function:

``child_state_choices``
    The state-choice whose beginning-of-period wealth we are computing. In the
    solve path this is the *child*. Its own identity is what the transition
    function sees -- including ``lagged_choice``, which *is* the parent's choice.

``representative_parent_state_choice_vec``
    A representative *parent* state-choice, used only to select whose continuous
    grid supplies the values fed through the transition. Once grids are
    state-choice-specific these are not interchangeable: the transition depends on
    the child's identity, but the grid *values* must come from the parent that is
    actually transitioning.

Any one parent works as the representative, because
``check_continuous_grid_consistency_across_shared_children`` (run once at
model-build time) already guarantees every parent sharing a child agrees on its
own grid. The representative is picked in
``pre_processing/batches/child_state_dedup.py``.

Two evaluation granularities
-----------------------------

The transition into a child normally does not depend on the child's own choice,
only on its state. So unless the budget equation declares ``choice``, every
state-choice sharing a child state would compute a bit-identical transition, and
evaluating per state-choice does redundant work proportional to the number of
choices.

``dcegm`` therefore has two implementations, dispatched between by
``calc_law_of_motion`` -- the single entry point every caller uses, so the
granularity decision lives in one place rather than being repeated per call site:

.. list-table::
   :header-rows: 1
   :widths: 32 30 38

   * - Function
     - Evaluates once per
     - Used when
   * - ``calc_law_of_motion_for_state_choices``
     - child **state-choice**
     - the budget equation declares ``choice``
   * - ``calc_law_of_motion_for_child_states``
     - unique child **state**
     - it does not (the common case)

.. note::

   The granularity is chosen for both transitions at once, on the budget equation
   alone. Since ``next_period_continuous_state`` can never declare ``choice``, a
   model with a choice-dependent budget recomputes a bit-identical continuous
   state for each of a child's choices. Splitting the two apart would mean
   carrying both sets of dedup arrays and a gather in between, for a quantity of
   shape ``(n_rows, n_combos)`` against wealth's
   ``(n_rows, n_combos, n_savings, n_shocks)`` -- a small fraction of the work, so
   the two are kept in one pass.

Both call sites -- ``interpolate_value_and_marg_util`` for the main backward
induction, and ``solve_final_period`` for the terminal period -- go through
``calc_law_of_motion`` and so take the same shortcut. The final period needs its
own dedup arrays, since it is not built by ``compute_child_dedup_for_batch``;
those are constructed in ``last_two_periods.py``
(``unique_final_period_states``,
``representative_second_last_period_parent_idx_per_final_state``,
``state_row_for_final_period_state_choice``) by deduplicating
``parent_states_final_period``.

Crucially these are not two implementations of the transition math.
``calc_law_of_motion_for_child_states`` *calls*
``calc_law_of_motion_for_state_choices`` with deduplicated child states instead of
state-choices, then gathers the per-state result back out to state-choice
granularity:

.. code-block:: python

    law_of_motion_per_state = calc_law_of_motion_for_state_choices(
        child_state_choices=child_states,               # deduplicated
        representative_parent_state_choice_vec=representative_parent_state_choices,
        ...
    )

    return {
        "continuous_states": {
            name: jnp.take(grid, state_row_for_state_choice, axis=0)
            for name, grid in law_of_motion_per_state["continuous_states"].items()
        },
        "assets_begin_of_period": jnp.take(
            law_of_motion_per_state["assets_begin_of_period"],
            state_row_for_state_choice, axis=0,
        ),
    }

``state_row_for_state_choice`` maps each child state-choice back to its row in the
deduplicated child states. It is built in ``child_state_dedup.py`` alongside the
dedup arrays that already existed, and follows the same gather pattern
``calculate_candidate_solutions_from_euler_equation`` uses one stage later for
``marg_util_next``.

How the choice is made
-----------------------

By signature inspection, once at model-build time -- the same approach
``taste_shock_scale_is_scalar`` uses:

.. code-block:: python

    # process_model_functions.py
    def _declares_choice(func):
        return "choice" in set(inspect.signature(func).parameters)

    budget_depends_on_choice = _declares_choice(budget_constraint)

The result is stored as ``model_funcs["budget_depends_on_choice"]``. Inspection
happens on the *user's* function, before
``determine_function_arguments_and_partial_model_specs`` wraps it -- the wrapper's
``**kwargs`` signature would hide the real parameter names.

The same inspection is used in the opposite direction for
``next_period_continuous_state``: ``_check_choice_not_declared`` raises a
``ValueError`` if it declares ``choice``, so the only function that can put the
solve on the per-state-choice path is the budget equation.

All shipped toy models (``with_cont_exp``, ``with_exp``, ``dcegm_paper``) report
``False``, so the deduplicated path is what the test suite actually exercises;
``test_default_toy_models_take_the_state_level_fast_path`` asserts this so it
cannot silently regress.

.. _choice_dependent_budget:

Writing a choice-dependent budget equation
--------------------------------------------

Declare ``choice`` as an argument and it will be passed through:

.. code-block:: python

    def budget_constraint(
        period, lagged_choice, choice, asset_end_of_previous_period,
        income_shock, params, model_specs,
    ):
        wealth = ...  # usual computation
        # e.g. a choice-specific fixed cost deducted at the start of the period
        return wealth - model_specs["entry_cost"] * (choice == 1)

Functions that do not declare ``choice`` are unaffected:
``determine_function_arguments_and_partial_model_specs`` filters kwargs down to
each function's own signature, so the extra key is simply ignored.

.. important::

   **A choice-dependent budget equation must return pre-consumption wealth.**

   The ``choice`` seen here is the agent's own choice in the period whose wealth
   is being computed, so the timing is unusual: wealth is resolved *after* the
   discrete choice but *before* the continuous consumption choice. ``dcegm``
   builds the agent's decision on exactly that reading -- it evaluates the budget
   equation once per choice, and the agent then compares each choice's value at
   that choice's own wealth, consuming out of it. Anything the budget equation
   subtracts is therefore money the agent never gets to consume; anything it adds
   is consumable in the same period.

   This is why only wealth works this way. A choice-specific cost or transfer is
   naturally levied at the moment of choosing, and consumption adjusts to it
   within the period. Every other state variable is fixed at the start of the
   period, before the choice, so ``next_period_continuous_state`` is evaluated at
   the *end* of the previous period and is not offered ``choice`` at all. The
   parent's choice is available separately, and always has been, as the child's
   ``lagged_choice``.

.. note::

   This did not work before the granularity split. Two places stripped ``choice``
   before calling user functions -- ``state_vec.pop("choice", None)`` in
   ``law_of_motion.py`` and ``state_vec.pop("choice")`` in ``final_periods.py`` --
   so a budget equation declaring ``choice`` raised ``KeyError: 'choice'``. Both
   pops were removable: the signature filter already does that job.

Simulation follows the same split. ``assets_begin_of_period_for_each_choice``
(``simulation/sim_utils.py``) evaluates the budget equation for every choice at
the start of a period, before the choice is drawn, and the realized wealth is the
column belonging to the choice actually made. The additional continuous states are
settled one step earlier, in ``transition_to_next_period`` at the end of the
previous period, and so carry no choice axis at all.

Cost
-----

Deduplication removes a factor of (choices per child state) from the *entire*
four-dimensional product above, not from one axis in isolation -- so it is a
straight ``1/n_choices`` reduction in how often the budget equation is evaluated,
independent of how large the combo/savings/shock axes are. What it pays instead is
a gather: pure data movement, no FLOPs beyond address computation.

That trade should favour deduplication when the budget equation is
arithmetic-heavy, and matters little when it is trivial (``R * savings + income``
fuses into the surrounding kernel at negligible cost, and the gather becomes pure
overhead). Since the fallback only triggers on an explicit ``choice`` argument,
models with a cheap choice-independent budget still get the dedup -- harmless,
just not a large win.

What is not measured
---------------------

The paragraph above is reasoning, not measurement. Nothing in ``dcegm`` currently
profiles this, and the crossover point -- how many FLOPs the budget equation needs
before dedup beats the gather -- is unknown. Confirming it would need
``jax.block_until_ready()`` timing or the XLA profiler, comparing a model with a
genuinely heavy budget equation and a realistic choice count under both paths
(forceable by adding an unused ``choice`` argument, as
``test_state_specific_grids_law_of_motion.py`` does). The same caveat the
:ref:`batching_internals` page applies to its own performance claims applies here.

Correctness testing
--------------------

The central guarantee is that the two granularities agree.
``tests/test_state_specific_grids_law_of_motion.py`` establishes this
(``test_state_level_and_state_choice_level_paths_agree_bit_for_bit``) by solving
the same economics twice -- once normally, once with a budget equation that declares an
otherwise-unused ``choice`` to force the per-state-choice path -- and requiring
bit-for-bit identical ``value``, ``policy`` and ``endog_grid``. A wrong
``state_row_for_state_choice`` mapping would surface there immediately, as
children would be fed another state's transition.

That test is paired with a sensitivity check
(``test_genuinely_choice_dependent_budget_differs_from_choice_free_one``): a budget
equation that actually *uses* ``choice`` must produce a different solution. Without
it, the equivalence test would still pass if ``choice`` were being silently
dropped again -- the exact regression it is there to catch.

The restriction on ``next_period_continuous_state`` has its own test,
``test_continuous_state_transition_declaring_choice_is_rejected`` in
``tests/test_state_specific_grids_reference.py``: a transition declaring
``choice`` must raise at model build, not be silently ignored.
