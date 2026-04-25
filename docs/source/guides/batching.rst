.. _batching_guide:

Batching Strategy and Segmentation
=================================

The backward induction in ``dcegm`` is solved in batches. This is a computational detail to make array shapes compatible with fast JAX scans while preserving the model logic.

Why batching exists
-------------------

The number of feasible state-choice combinations usually changes over the life cycle. However, vectorized scan steps work best with equal leading dimensions. Batching groups state-choice rows into equal-sized chunks so each scan step can run with fixed shapes.

Two batching modes
------------------

``dcegm`` supports two batching modes:

- ``largest_block``:

  - Finds large dependency-safe batches.
  - Typically yields fewer and larger batches.
  - Good default for smooth state-choice profiles.

- ``period_max``:

  - Uses one batch per period within a segment.
  - Pads smaller period batches to the segment-specific maximum number of state choices per period.
  - Useful when state-choice counts vary strongly by period.
  - **Padding rule**: If a period has fewer state choices than the segment maximum, the batch is padded with a valid dummy state-choice index from the same batch (deterministically the first one). This keeps shapes aligned and does not change the solution logic.

Segmenting the horizon
----------------------

Use ``min_period_batch_segments`` to split the pre-terminal part of the horizon into segments.

- Without segmentation:

  - ``batch_mode`` must be a single string.

- With segmentation:

  - ``batch_mode`` can be a string (reused for all segments), or
  - ``batch_mode`` can be a list with one entry per segment.

The number of segments is ``len(min_period_batch_segments) + 1``.

Valid strings are ``"largest_block"`` and ``"period_max"``.

Examples
~~~~~~~~

No segmentation:

.. code-block:: python

    model_config = {
        "n_periods": 20,
        "choices": np.arange(3, dtype=int),
        "continuous_states": {"assets_end_of_period": np.linspace(0, 100, 200)},
        "n_quad_points": 5,
        "batch_mode": "period_max",
    }

With segmentation:

.. code-block:: python

    model_config = {
        "n_periods": 20,
        "choices": np.arange(3, dtype=int),
        "continuous_states": {"assets_end_of_period": np.linspace(0, 100, 200)},
        "n_quad_points": 5,
        "min_period_batch_segments": [8, 14],
        "batch_mode": ["period_max", "largest_block", "period_max"],
    }

Tipp: Use ``get_n_state_choices_per_period`` to choose segments
----------------------------------------------------------------

To determine sensible segments for batching, inspect the number of state-choice combinations per period.

.. code-block:: python

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
        state_space_functions=state_space_functions,
        stochastic_states_transitions=stochastic_states_transitions,
    )

    n_state_choices = model.get_n_state_choices_per_period()
    print(n_state_choices)

This series can be used to detect structural breaks in complexity. Typical heuristics are:

- Keep periods with similar counts in one segment.
- Split where there are abrupt jumps/drops.
- Use ``period_max`` in highly uneven segments.
- Keep ``largest_block`` in smoother segments.

Example: experience growth and retirement regimes
-------------------------------------------------

Consider a model with a discrete experience state where:

- choice 0: no work, experience unchanged,
- choice 1: regular work, experience increases by 1,
- choice 2: intensive work, experience increases by 2,
- choice 3: retirement.

Suppose retirement becomes available from period 8, and is mandatory from period 14.

.. code-block:: python

    def choice_set(period, lagged_choice):
        if period >= 14:
            return np.array([3], dtype=int)              # mandatory retirement
        if period >= 8:
            return np.array([0, 1, 2, 3], dtype=int)     # retirement becomes available
        return np.array([0, 1, 2], dtype=int)

    def next_period_deterministic_state(period, choice, experience):
        if choice == 1:
            experience_next = experience + 1
        elif choice == 2:
            experience_next = experience + 2
        else:
            experience_next = experience
        return {
            "period": period + 1,
            "lagged_choice": choice,
            "experience": experience_next,
        }

In this setup you often see:

- gradual growth in state-choice counts early on,
- a jump when retirement becomes optional,
- a drop when retirement becomes mandatory.

This pattern is a good reason to separate segments around the two regime changes:

.. code-block:: python

    model_config["min_period_batch_segments"] = [8, 14]
    model_config["batch_mode"] = ["period_max", "largest_block", "period_max"]

We suggest testing different segmentation choices to determine the fastest solution for your model.
