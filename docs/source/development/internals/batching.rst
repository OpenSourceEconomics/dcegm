.. _batching_internals:

Batching: Internals, Alternatives, and Open Questions
=======================================================

.. warning::

   This page is AI-generated (drafted with Claude, based on reading the source and a
   design discussion, without access to GPU hardware to verify the performance and
   memory claims empirically). It is intended as internal documentation for
   contributors and as context for other AI coding agents working on this codebase,
   not as a peer-reviewed reference. Claims marked as unverified or plausible in the
   text below have not been checked against real hardware; verify before relying on
   them for a decision.

.. note::

   This page is for contributors working on the solver itself. If you only want to
   *configure* batching for your model, see the :ref:`batching_guide` in the
   background section instead. This page explains the mechanism behind that
   configuration, discusses algorithmic alternatives, and relates the design to the
   wider computational-science literature on batching irregular workloads.

Why batching exists
--------------------

Backward induction in ``dcegm`` solves one period at a time, from the terminal
period backward, and each period's continuation values depend on the (already
solved) next period. The number of feasible state-choice combinations changes over
the life cycle -- e.g. a choice becomes unavailable, or a deterministic state stops
being reachable -- so the *natural* unit of work (one period) does not have a fixed
size.

That would not matter for a plain Python loop, but the backward induction loop in
``dcegm`` is implemented as a single :func:`jax.lax.scan` per segment
(:func:`dcegm.backward_induction.backward_induction`). ``lax.scan`` stacks its
per-step inputs into one array and traces the step function *once*, so every step
must consume an array slice of the *same shape*. Batching exists to reconcile these
two facts: it repackages a horizon of unevenly-sized periods into a sequence of
equal-sized chunks that ``lax.scan`` can iterate over, while preserving the
dependency order backward induction requires (a state-choice's children must be
solved in a strictly earlier batch).

How it is currently done
-------------------------

All batching logic lives under ``src/dcegm/pre_processing/batches/``. The user-facing
entry point is :func:`dcegm.pre_processing.batches.batch_creation.create_batches_and_information`,
which splits the horizon into one or more *segments* (see below), and delegates the
construction of batches within each segment to
:func:`dcegm.pre_processing.batches.single_segment.create_single_segment_of_batches`.

Within a segment, ``dcegm`` supports two modes:

``largest_block``
    Implemented in
    :func:`dcegm.pre_processing.batches.algo_batch_size.determine_optimal_batch_size`.
    All eligible state-choices in the segment are sorted once, ascending, by the
    minimum raw state-choice index of their child states. This is the only ordering
    step -- see the open question below. The sorted sequence is then reversed and
    split into contiguous chunks of size ``current_batch_size``, starting from
    ``current_batch_size = size_last_period`` (the number of state-choices in the
    segment's last period).

    Each candidate size is checked for validity: for every chunk, the maximum raw
    state-choice index in the chunk must be *smaller* than the minimum raw
    state-choice index among that chunk's required children. If this fails for any
    chunk, ``current_batch_size`` is shrunk by 2% (``current_batch_size = int
    (current_batch_size * 0.98)``) and *every* chunk is re-validated from scratch.
    This repeats until a valid uniform size is found. In pseudocode:

    .. code-block:: text

        sort state-choices ascending by min(child state-choice index)
        current_batch_size = size_last_period
        loop:
            chunks = split(reverse(sorted_state_choices), current_batch_size)
            if all(chunk.max_index < min(child_indices(chunk)) for chunk in chunks):
                return chunks
            current_batch_size = int(current_batch_size * 0.98)

    This is a correctness-constrained search for a large uniform batch size, not a
    padding-vs-waste tradeoff: because ``lax.scan`` reuses one compiled step
    regardless of how many iterations it runs, a larger valid batch size is always
    preferable within a segment (fewer scan steps, same compiled kernel, no extra
    compile cost). See :ref:`batching_alternatives` for a cheaper way to find it.

``period_max``
    Implemented in ``determine_period_max_batch_size`` in the same module. Each
    period within the segment becomes exactly one batch. Batches are padded to the
    segment's largest per-period state-choice count with a deterministic dummy
    state-choice index (the first valid one in the same batch), so padding never
    changes the solution -- it only wastes compute on the padded slots.

Segmenting the horizon
~~~~~~~~~~~~~~~~~~~~~~~

``min_period_batch_segments`` (handled in ``batch_creation.py``) splits the horizon
into multiple segments *before* either mode above runs. In
:func:`dcegm.backward_induction.backward_induction`, each segment gets its own
``jax.lax.scan`` call, via a plain Python ``for id_segment in range(n_segments):``
loop -- ``n_segments`` is a static Python int fixed at model-setup time, not a traced
value. This is the mechanism that lets different parts of the life cycle use
different batch modes or (implicitly, via ``largest_block``) different uniform batch
sizes: a single global batch size is bottlenecked by the most dependency-dense
region of the horizon, and segmenting lets the rest of the horizon avoid that
bottleneck.

What that costs depends on *how* ``backward_induction`` is called, and this matters
because the two call sites behave differently:

- :meth:`~dcegm.interfaces.model_class.setup_model.solve` calls it eagerly, with no
  enclosing ``jax.jit``. Each segment's ``lax.scan`` is then traced and compiled as
  its own separate XLA computation, dispatched one after another.
- :meth:`~dcegm.interfaces.model_class.setup_model.get_solve_func` (the recommended
  path for repeated solves, e.g. inside :mod:`dcegm.likelihood`, which follows the
  same pattern) wraps the *whole* ``backward_induction`` call -- Python loop included
  -- inside one outer ``jax.jit``. Because ``n_segments`` is static, tracing unrolls
  that loop at trace time into :math:`n_{\text{segments}}` sequential ``lax.scan``
  primitives inside a single jaxpr, which XLA then compiles as **one** executable
  containing :math:`n_{\text{segments}}` back-to-back scan sub-computations -- not as
  separate executables.

Wrapping everything in one outer ``jax.jit`` is not free (see the warning below), so
it is worth being explicit about what it buys, since compiling each segment
separately would also get you the standard JAX benefit of "compile once on the first
call, reuse the executable for every later ``params``" -- that part is not what
distinguishes the two designs. What the *single* enclosing jit adds on top:

- **No host round-trip between segments.** Under eager ``.solve()``, each segment's
  ``lax.scan`` is a separate dispatch: Python-side argument/pytree handling and a
  host-device synchronization boundary between segments. On GPU, dispatch and kernel
  launch latency is frequently the actual bottleneck for a sequence of small-to-medium
  ops, more so than raw compute throughput -- collapsing :math:`n_{\text{segments}}`
  dispatches into one removes that overhead entirely for everything after the first
  compile.
- **Whole-program scheduling.** XLA's scheduler and buffer-assignment see the entire
  unrolled sequence at once and can reorder or overlap work across segment boundaries
  where data dependencies allow. :math:`n_{\text{segments}}` separately-compiled
  programs are each optimized in isolation, with zero visibility past their own jit
  boundary.

.. warning::

   Even under the single-``jax.jit`` path, more segments plausibly still means more
   memory, but the mechanism is different from "many separate compiled programs
   competing for device memory":

   - **Compile-time/host cost (the solid part).** More segments means a larger
     unrolled jaxpr/HLO program for XLA to schedule, buffer-assign, and (if enabled)
     autotune. Compile time and the host RAM used *during* compilation both grow with
     program size -- a general, well-documented property of XLA for large unrolled
     graphs, not specific to ``dcegm``.
   - **Device/runtime cost (plausible, not verified here).** XLA's buffer-assignment
     does whole-program liveness analysis, so in principle it can still reuse buffers
     *across* segment boundaries within that one compiled program -- the earlier
     framing of "no reuse across separate executables" does not apply here, because
     there is only one executable. What can still block reuse is a *shape change* at
     a segment boundary: the ``value``/``policy``/``endog_grid`` arrays threaded as
     the scan carry from one segment into the next only alias cleanly in place when
     shapes match, and adjacent segments deliberately using different batch sizes or
     modes is the entire point of segmenting. So the more precise (but unverified)
     claim is that the number of *segment boundaries* -- not the number of segments
     or executables as such -- is what may force extra allocations instead of
     in-place reuse.
   - **Batch-index metadata.** ``batch_info`` holds every segment's index arrays
     (``batches_state_choice_idx`` and friends, built by
     ``prepare_and_align_batch_arrays``) simultaneously, since all segments are
     constructed up front before ``backward_induction`` runs. More segments means
     more such arrays resident at once, though this is likely small next to the
     solution containers.

   None of this is currently measured in ``dcegm``. Confirming it, and by how much,
   would need ``jax.devices()[0].memory_stats()["peak_bytes_in_use"]`` (with
   ``XLA_PYTHON_CLIENT_PREALLOCATE=false``, since JAX preallocates most GPU memory by
   default and would otherwise hide the effect) compared across a fixed model solved
   through :meth:`~dcegm.interfaces.model_class.setup_model.get_solve_func` with a
   varying number of segments. See the cost model below, which currently only
   accounts for compile *time*, not memory.

Today, segment boundaries and per-segment modes are chosen by hand (see the
:ref:`batching_guide` for the recommended workflow using
``get_n_state_choices_per_period``). This manual step is one of the main things
:ref:`batching_alternatives` below tries to replace.

Relation to the wider literature
---------------------------------

``dcegm``'s batching is an instance of a problem that shows up, with different names,
across computational science whenever irregular workloads need to run on hardware
that wants fixed shapes. Three literatures map onto the two mechanisms above:

- **Dependency-safe batch construction** (``largest_block``'s validity check) is what
  parallel computing calls *wavefront* or *level-scheduled* parallelism: group a DAG's
  nodes into levels such that everything in a level depends only on already-computed
  levels. It is classically used for sequence alignment and PDE stencils on GPUs.

  - Kartik Hegde et al. `Memory-Optimized Wavefront Parallelism on GPUs <https://link.springer.com/article/10.1007/s10766-020-00658-y>`_. *International Journal of Parallel Programming* (2020).
  - `Taskflow: Wavefront Parallelism <https://taskflow.github.io/taskflow/wavefront.html>`_ (pedagogical overview of the pattern).

- **Sizing batches under a hard uniformity constraint** is the classical *multiprocessor
  scheduling* / *bin-packing* problem: assign items to a fixed number of equal-capacity
  bins to minimize the number of bins (equivalently, maximize bin size). The Longest
  Processing Time (LPT) heuristic gives a 4/3-approximation guarantee.

  - E. G. Coffman, M. R. Garey, D. S. Johnson (1978). `An Application of Bin-Packing to Multiprocessor Scheduling <https://epubs.siam.org/doi/10.1137/0207001>`_. *SIAM Journal on Computing*.

- **Padding variable-size groups to a fixed shape** (``period_max``) is the same
  mechanism as sequence bucketing in batched ML inference, including the same
  padding-vs-compute-waste tradeoff and the same underlying cause: neither XLA nor
  most deep-learning runtimes have first-class ragged-array support.

  - `Continuous batching from first principles <https://huggingface.co/blog/continuous_batching>`_. Hugging Face (2025).
  - `Support for ragged arrays, like torch.nested <https://github.com/jax-ml/jax/issues/17863>`_. jax-ml/jax issue tracker.

- For the domain itself, GPU-batched Bellman backward induction shows up directly in
  recent economics/OR work, useful as motivation and cross-validation of the general
  approach rather than as an algorithmic source:

  - `GPU-Accelerated Dynamic Programming for Multistage Stochastic Energy Storage Arbitrage <https://arxiv.org/abs/2511.15629>`_.
  - `Structural Reinforcement Learning for Heterogeneous Agent Macroeconomics <https://arxiv.org/html/2512.18892>`_. Moll et al.

.. _batching_alternatives:

Alternatives worth considering
--------------------------------

Splitting the current design into its two decisions clarifies which improvements are
"free" and which involve a genuine, hardware-dependent tradeoff.

Within a segment: replace the multiplicative-decrease search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The validity check used by ``largest_block`` is monotonic in batch size: a larger
uniform batch size spans a wider range of raw indices, which can only make the
"children already solved" check harder to satisfy, never easier. That means the set
of valid batch sizes is downward-closed, and the true maximum can be found by
**binary search** instead of shrinking by 2% and re-validating every chunk from
scratch at every step:

.. code-block:: text

    function max_valid_batch_size(state_choices_sorted_by_child_idx):
        lo, hi = 1, len(state_choices_sorted_by_child_idx)
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(state_choices_sorted_by_child_idx, mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

``feasible`` is the same per-chunk check already implemented today; only the search
strategy changes, from O(chunks-per-attempt :math:`\times` attempts-to-converge) with
an arbitrary 2%-undershoot, to O(chunks-per-attempt :math:`\times \log n`) landing on
the true maximum. This is a strict improvement, not a hardware-dependent tuning
choice -- it should be paired with an equivalence test (brute-force linear scan vs.
binary search on a couple of existing toy models) before being trusted, both to
confirm monotonicity holds in practice and following the project's convention of
pairing solver-internals refactors with a golden-value/equivalence test.

Across segments: a calibrated cost model instead of a manual knob
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choosing how many segments to use, and where to split them, *is* genuinely
hardware- and workload-specific, because each extra segment buys a larger batch size
for the rest of the horizon at the cost of one extra XLA compile. Formalizing this
needs two device- and workload-specific constants:

- :math:`C_{\text{compile}}`: fixed cost of one additional segment (a separate
  ``lax.scan`` compilation). Matters most for a single ``solve()`` call; matters much
  less inside an estimation loop that reuses the compiled function across many
  parameter draws.
- :math:`C_{\text{step}}`: marginal cost per scan iteration (kernel dispatch plus
  actual FLOPs). Shrinks, relative to a fixed batch size, the more GPU-bound the
  workload is.

Given those, choosing segment boundaries becomes a shortest-path / dynamic-programming
problem over candidate splits of the period axis:

.. code-block:: text

    function best_segmentation(periods, C_compile, C_step):
        n = len(periods)
        cost = [0] + [infinity] * n
        split_at = [None] * (n + 1)
        for j in 1..n:
            for i in 0..j-1:
                B = max_valid_batch_size(periods[i:j])
                n_scan_steps = ceil(size(periods[i:j]) / B)
                segment_cost = C_compile + n_scan_steps * C_step
                if cost[i] + segment_cost < cost[j]:
                    cost[j] = cost[i] + segment_cost
                    split_at[j] = i
        return backtrack(split_at)

:math:`C_{\text{compile}}` and :math:`C_{\text{step}}` should be measured on the
target device (time a representative ``lax.scan`` compile, and a few iterations at
different batch sizes to fit the marginal per-step cost) rather than guessed. With
:math:`C_{\text{compile}} \to 0` the optimum drifts toward many small segments (one
period per block, minimal batch-size compromise); with a large
:math:`C_{\text{compile}}` it collapses to a single segment (today's default when
``min_period_batch_segments`` is not set). This also explains *why* segmenting ever
helps: a single global batch size is bottlenecked by the horizon's most
dependency-dense region, and paying for an extra compile lets the rest of the horizon
escape that bottleneck.

As modeled above, :math:`C_{\text{compile}}` only represents compile *time* -- but,
per the warning above, each extra segment also has a device-memory cost from holding
more separately-compiled executables at once. Time is a cost worth trading off
against faster steps; memory is closer to a hard constraint, since exceeding it means
the solve fails outright rather than merely running slower. A more faithful version
of this DP would therefore track a *memory budget* alongside the additive time cost
(minimize total time subject to peak memory across live segments staying under the
device limit) rather than folding both into one scalar cost -- but that requires
measuring the per-segment memory cost first, which nothing in ``dcegm`` does today.

A further-out alternative for ``period_max``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``period_max``'s padding could in principle be avoided altogether by flattening all
periods into one array and using masked/segmented reductions (JAX's
``jax.ops.segment_sum``-style primitives) instead of materializing padded batches --
the same idea as "continuous" or "packed" batching in ML serving. The real tradeoff
is that if batch shape varies per scan step, ``lax.scan`` can no longer reuse one
compiled step, trading padding waste against recompilation cost. Whether that trade
is worth it needs profiling on representative model sizes; it is not a clear win
either way and is listed here as a direction, not a recommendation.

Open question: does the state-choice ordering matter?
--------------------------------------------------------

Before splitting into chunks, ``largest_block`` sorts the eligible state-choices by a
single, fixed key -- ascending minimum raw index of their child states -- and never
searches over alternative orderings. This ordering determines how tightly packed a
state-choice's dependencies are relative to its own position, which in turn
determines how large a uniform batch can get before the validity check fails.

It is currently unknown whether this particular ordering is a good one, or even a
reasonable default, relative to the alternatives. The closest known analogue is
**bandwidth minimization / fill-reducing reordering** in sparse linear algebra
(reverse Cuthill-McKee, nested dissection), where the entire point of choosing an
ordering is to keep each row's dependencies as close as possible to the row itself,
which directly shrinks the "distance" a dependency-safe block needs to span. If the
same intuition transfers here, a bandwidth-minimizing ordering might permit
substantially larger valid batch sizes than the current ascending-child-index sort
achieves -- but this has not been tested, and it is also possible that the current
sort is already close to optimal for typical ``dcegm`` life-cycle structures (where
dependencies are inherently local, since a state-choice's children mostly live in the
immediately next period). Resolving this would need a small experiment: compare the
maximum valid batch size (via :ref:`batching_alternatives`'s binary search) under the
current sort against a bandwidth-minimizing reordering, on a model with an irregular
enough state space to make a difference.
