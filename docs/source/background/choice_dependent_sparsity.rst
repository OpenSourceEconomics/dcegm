.. _choice_dependent_sparsity:

Choice-Dependent Sparsity
==========================

.. warning::

   This page is AI-drafted (with Claude, based on reading the source and a design
   discussion with a maintainer). It documents current behaviour as of writing;
   verify against the source if something looks off.

.. note::

   This page assumes you have read :ref:`sparsity_conditions`, which covers the
   basics of ``sparsity_condition``. This page answers a narrower question: how to
   express restrictions that depend on a *choice*.

The short answer
-----------------

"Sparsity that depends on choice" is two different requirements wearing one name,
and ``dcegm`` already supports both -- through two different functions. Which one
you need depends on *which* choice you mean:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - You want to restrict...
     - Use
     - Because
   * - based on the choice made **last** period
     - ``sparsity_condition``, via the ``lagged_choice`` argument
     - ``lagged_choice`` is an ordinary discrete state variable
   * - which choices are available **in** a state
     - ``state_specific_choice_set``
     - this *is* the (state, choice)-level sparsity mechanism

There is deliberately no ``choice`` argument to ``sparsity_condition``. The next
section explains why, and the two after that give the recipe for each case.

Why ``sparsity_condition`` cannot take ``choice``
--------------------------------------------------

Two reasons, one conceptual and one structural.

**Conceptually**, a state's existence cannot depend on a choice made *from* that
state. The agent is in the state first and chooses second; asking "is this state
valid given the choice?" inverts the timing. What people usually mean when they
reach for this is one of the two supported cases above.

**Structurally**, ``sparsity_condition`` is evaluated in
:func:`dcegm.pre_processing.model_structure.state_space.create_state_space`, which
runs *before* the state-choice space exists:

.. code-block:: python

    # state_space.py, inside the period / endog-state / lagged-choice loop
    state_dict = {
        discrete_states_names[i]: state_value
        for i, state_value in enumerate(state)
    }
    sparsity_output = sparsity_condition(**state_dict)

``state_dict`` is built purely from discrete *state* variables. The state-choice
space is only assembled afterwards, in
``create_state_choice_space_and_child_state_mapping``, which loops over the
already-filtered valid states and calls ``state_specific_choice_set`` for each.
So the two functions run at different stages by construction, and each sees
exactly the information available at its stage.

Case 1: restricting on last period's choice
--------------------------------------------

This needs nothing special -- ``lagged_choice`` is a state variable like any
other, so just declare it as an argument. The shipped ``with_exp`` toy model
(``toy_models/cons_ret_model_with_exp/state_space_objects.py``) does exactly
this:

.. code-block:: python

    def sparsity_condition(period, experience, lagged_choice, model_specs):
        max_exp_period = period + model_specs["max_init_experience"]
        ...
        # If experience is the maximum for this period, you must have worked
        if (experience == max_exp_period) & (lagged_choice == 1):
            return False
        # Retirement is absorbing: if you worked last period, experience must
        # be at least the period count
        elif (lagged_choice == 0) & (experience < period):
            return False
        else:
            return True

Both branches are choice-dependent restrictions in the everyday sense -- they
just express it through the state variable that *records* the previous choice.

Case 2: restricting which choices a state offers
-------------------------------------------------

This is ``state_specific_choice_set``, which returns the feasible choice set for
a given state. It is called once per valid state during state-choice space
construction:

.. code-block:: python

    # state_choice_space.py
    feasible_choice_set = state_specific_choice_set(**this_period_state)

    for choice in feasible_choice_set:
        state_choice_space_raw[idx, :-1] = state_vec
        state_choice_space_raw[idx, -1] = choice
        ...

Any (state, choice) pair you omit from the returned set simply never enters the
state-choice space -- no solution container row, no EGM candidate, no upper
envelope call. That is precisely "sparsity at the choice level".

The shipped ``dcegm_paper`` model
(``toy_models/cons_ret_model_dcegm_paper/state_space_objects.py``) uses it to make
retirement absorbing:

.. code-block:: python

    def get_state_specific_feasible_choice_set(lagged_choice, model_specs):
        n_choices = model_specs["n_choices"]

        # Once the agent chooses retirement, she can only choose retirement
        # thereafter. Hence, retirement is an absorbing state.
        if lagged_choice == 1:
            feasible_choice_set = np.array([1])
        else:
            feasible_choice_set = np.arange(n_choices)

        return feasible_choice_set

.. warning::

   Choice encodings are per-model, not a package-wide convention: in
   ``dcegm_paper`` above ``lagged_choice == 1`` is retirement, while in the
   ``with_exp`` sparsity condition quoted earlier ``lagged_choice == 0`` is
   working. Read the model you are editing rather than carrying an encoding over
   from an example.

The two must agree
-------------------

This is where most real bugs live. The two mechanisms are checked against each
other at model-build time, because they jointly have to produce a closed
transition system: every feasible (state, choice) must lead somewhere valid, and
every valid non-initial state should be reachable.

**Check 1 -- every state-choice needs a valid child** (raises ``ValueError``, in
``test_child_state_mapping``):

.. code-block:: text

    Some state-choice combinations have invalid child states. Please update
    accordingly the deterministic law of motion or the proxy function.

You get this when ``state_specific_choice_set`` offers a choice whose implied
child state (via ``next_period_deterministic_state``) was filtered out by
``sparsity_condition``. Fix by either removing that choice from the choice set,
loosening the sparsity condition, or proxying the child to a valid state (see
below).

**Check 2 -- every valid state should be reachable** (raises ``UserWarning``):

.. code-block:: text

    Some states are not child states of any state-choice combination or
    stochastic transition. Please revisit the sparsity condition.

You get this when ``sparsity_condition`` admits a state that no feasible
(state, choice) transitions into. It is a warning rather than an error because
an unreachable state is wasteful, not incorrect -- it is solved and stored, but
never read. Still worth fixing: it inflates every solution container.

A useful way to hold the two checks in mind: Check 1 says your choice set is too
*permissive* relative to your sparsity condition; Check 2 says your sparsity
condition is too permissive relative to your choice set.

The stochastic-state exception: proxying
------------------------------------------

There is one case where you cannot simply delete an invalid state: when it is
invalid because of a *stochastic* state component. Every state-choice is assumed
to have exactly ``n_stochastic_states`` children, so a missing one would break
that shape invariant.

For these, ``sparsity_condition`` returns a **dictionary** instead of a boolean --
the valid state to use in place of the invalid one:

.. code-block:: python

    def sparsity_condition(period, lagged_choice, job_offer, survival, model_specs):
        ...
        # Dead agents receive no job offers; collapse the whole (dead, job_offer)
        # family onto the job_offer = 0 representative.
        job_offer_out = 0 if survival == 0 else job_offer

        return {
            "period": period,
            "lagged_choice": lagged_choice,
            "survival": survival,
            "job_offer": job_offer_out,
        }

Returning the *same* state you were handed means "this state is valid" -- which
is why a single ``return {...}`` can serve both roles, and why you rarely need to
mix ``return False`` and ``return {...}`` in one function.

Two rules the build enforces:

- The proxy target must itself be valid. If it is not, model setup fails with
  ``The state ... is used as a proxy state for the state ... However, the proxy
  state is also declared invalid by the sparsity condition.``
- Proxy dictionaries must contain exactly the discrete state names, all
  integer-valued -- checked in ``create_state_space``.

Debugging
----------

When the checks above fire and the cause is not obvious, get the full picture of
what the sparsity condition did to every candidate state:

.. code-block:: python

    from dcegm.pre_processing.setup_model import create_model_dict

    state_space_df = create_model_dict(
        model_config=model_config,
        model_specs=model_specs,
        debug_info="state_space_df",
        **model_funcs,
    )

This returns a ``pandas.DataFrame`` over the *unfiltered* cross-product state
space, with one row per candidate: every discrete state variable as a column,
plus ``is_valid``, ``is_proxied``, and (when proxies exist) ``idxs_proxied_to``.
Filtering it to the state named in the error message usually localises the
problem in one step -- it is built by ``create_state_space(..., debugging=True)``,
so it reflects exactly the decisions the real build made.

.. warning::

   Use ``create_model_dict`` for this, not ``dcegm.setup_model``. ``setup_model``
   is a class whose ``__init__`` immediately indexes ``model_dict["model_config"]``,
   so passing ``debug_info="state_space_df"`` there fails with
   ``KeyError: 'model_config'`` -- the debug path returns a DataFrame, which has
   no such key.

Summary
--------

- Restricting on **last period's** choice: ``sparsity_condition``, via
  ``lagged_choice``. Nothing special required.
- Restricting **this period's** available choices: ``state_specific_choice_set``.
  This is the choice-level sparsity mechanism.
- ``sparsity_condition`` takes no ``choice`` argument, by design -- it runs before
  the state-choice space exists, and state validity logically precedes choice.
- Keep the two consistent: every feasible (state, choice) needs a valid child
  (``ValueError``), and every valid state should be reachable (``UserWarning``).
- Invalid-because-stochastic states get proxied to a valid representative rather
  than dropped, to preserve the ``n_stochastic_states``-children invariant.
