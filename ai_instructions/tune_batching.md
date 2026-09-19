# Tune the batching of a dcegm model

Pick `model_config["min_period_batch_segments"]` and `model_config["batch_mode"]` so
the backward induction runs as few `lax.scan` steps as the model allows, without
paying for it in padded work or device memory.

Do this when a solve is slower than its size suggests, when the model-setup log shows
a batch width far below the model's per-period state-choice counts, or after any
change to the state space (a new state, a changed sparsity condition, a different
number of periods) -- the right answer is a property of *this* model's per-period
sizes and does not survive such a change.

Mechanism and design alternatives are documented in
`docs/source/development/internals/batching.rst`; configuration is in
`docs/source/background/batching.rst`. This file is the procedure.

## What sets the number of steps

Four facts decide everything below:

1. **A period's children are in the next period.** Two periods can therefore never
   share a batch, so the number of scan steps is at least the number of periods the
   scan covers. The last two periods are solved outside the scan
   (`solve_last_two_periods`), so the scan covers periods `0 .. n_periods - 3`, and

   > **the step floor is `n_periods - 2`.**

2. **`lax.scan` needs one batch width per segment.** Every step of a segment's scan
   consumes the same shape.

3. **`largest_block` is capped by the segment's *last* period.** Its first batch holds
   the state-choices whose children are already solved -- that is exactly the
   segment's last period. Anything wider drags in parents whose children sit in the
   same batch, which the validity check rejects. So one small period at the end of a
   segment throttles every other period in it. The search starts at that size and only
   ever shrinks, so the setup log's `start search` is an upper bound, not a guess.

4. **`period_max` gives one batch per period**, padded to the segment's largest
   period. It always hits the step floor for its segment; the cost is the padding.

So: **segment the horizon at the points where the per-period size changes, and use
`period_max`.** Padding is then small inside each segment, and you land on the floor.

## Procedure

### 1. Get the per-period state-choice counts

From a built model:

```python
model.get_n_state_choices_per_period()   # pandas Series, indexed by period
```

From a stored model pickle, without rebuilding:

```python
import pickle
import numpy as np

obj = pickle.load(open(MODEL_PICKLE, "rb"))
state_choice_space = np.asarray(obj["model_structure"]["state_choice_space"])
counts = np.bincount(state_choice_space[:, 0])          # column 0 is the period
previous = None
for period, count in enumerate(counts):
    if count != previous:                                # print only the steps
        print(f"period {period:3d}: {count:8,d}")
        previous = count
```

### 2. Read the profile as plateaus

Write down the periods where the count changes. Those change points are the segment
separators. Ignore the last two periods -- they are not in the scan.

A declining run (a count that falls a little every period) is one plateau, not
several: `period_max` pads it to its largest member, which is cheap as long as the
run is short. Split it only if the sweep in step 4 says the padding is material.

### 3. Propose separators

`min_period_batch_segments` is a list of **minimum periods, increasing**, and segments
are numbered **from the oldest**: `[29, 33, 42, 44]` on a 71-period model means

```
segment 0: periods 44-68     segment 3: periods 29-32
segment 1: periods 42-43     segment 4: periods  0-28
segment 2: periods 33-41
```

Entries must be increasing and below `n_periods - 4`. Start from the change points of
step 2, with `batch_mode="period_max"`.

### 4. Sweep the candidates offline

Do not guess which candidate wins -- run the real batching algorithm. It needs
`map_state_choice_to_child_states` and `map_state_choice_to_index`, which
`setup_model` deletes unless it is called with `debug_info="all"`, so build (or
rebuild) the model that way once.

```python
import io
import time
from contextlib import redirect_stdout

import jax
import numpy as np

from dcegm.pre_processing.batches.batch_creation import create_batches_and_information

# Build the model with debug_info="all" so the child maps survive, then:
model_structure = jax.tree.map(
    lambda x: np.asarray(x) if hasattr(x, "shape") else x, model.model_structure
)
assert "map_state_choice_to_child_states" in model_structure

CANDIDATES = [
    ([44], "largest_block"),            # whatever the model uses today, as baseline
    ([29, 33, 42, 44], "period_max"),   # the step-2 change points
    # ... further candidates
]

for separators, mode in CANDIDATES:
    with redirect_stdout(io.StringIO()):                 # setup logging is noisy
        info = create_batches_and_information(
            model_structure=model_structure,
            n_periods=n_periods,
            min_period_batch_segments=separators,
            batch_mode=mode,
        )
    steps = slots = 0
    for segment in range(info["n_segments"]):
        segment_info = info[f"batches_info_segment_{segment}"]
        n_batches, width = np.asarray(segment_info["batches_state_choice_idx"]).shape
        steps += n_batches + (0 if segment_info["batches_cover_all"] else 1)
        slots += n_batches * width
        print(f"    segment {segment}: {n_batches:4d} x {width:6,d}")
    print(f"### {separators} {mode} -> {steps} scan steps, {slots:,} slots")
```

`slots` (batches x width, summed) is the padded work: compare it against the model's
total state-choice count in the scan range to read off the padding overhead.

### 5. Choose

- **Steps first.** Going from hundreds of steps to the floor is the win; a solve
  stuck at a small batch width is dispatch-bound, and the per-step work is nearly
  free.
- **Then padding.** Among candidates that reach the floor, take the one with the
  fewest slots. Stop adding separators when the next one buys less than a few percent
  -- each costs compile time and a segment boundary (see the warning in the internals
  page).
- **`largest_block` is still the better mode** for a segment whose periods are all
  the same size *and* whose last period is one of them: it reaches the same width
  with no padding machinery. It is the wrong mode whenever a segment ends on a small
  period.

### 6. Verify on the device

The offline sweep proves the step count, not the speedup. Solve the model before and
after and compare wall time and peak memory. Two things to watch:

- **Peak memory goes up with batch width**, because the interpolated child arrays are
  `(batch width) x n_continuous_combinations x n_wealth x n_income_shocks`. If the
  new width no longer fits, `model_config["income_shock_batch_size"]` trades the
  income-shock axis back down (see the practitioner guide).
- **Compile time goes up with the number of segments**, since each is its own scan in
  the unrolled program. Matters for a single solve, much less inside an estimation
  loop that reuses the compiled function.

## Pitfalls

- A stored model pickle has `map_state_choice_to_child_states` and
  `map_state_choice_to_index` stripped. Rebuilding with `debug_info="all"` is the
  only way to run the sweep.
- `model.model_structure` holds jax arrays after setup; the batching code uses
  `np.take(..., mode=...)`, which dispatches to `jnp.take` and raises
  `NotImplementedError: The 'raise' mode to jnp.take is not supported`. Convert to
  numpy first, as in step 4.
- Separators are *minimum* periods and segment 0 is the oldest -- easy to get
  backwards.
- `batch_mode` may be one string for all segments or a list with one entry per
  segment (`n_separators + 1` entries, ordered oldest-first).
- Model variants that share a state space share a profile: check rather than assume,
  but a flag that only changes a transition function (not the state space) leaves the
  per-period counts identical, so one analysis covers both.
- Sub-models restricted to a type have their own profile. Re-run the procedure for
  each configuration that is actually solved.

## Worked example

A 71-period life-cycle model, 1,582,596 state-choices. Profile from step 1:

```
periods  0-28: 33,084     period     42: 20,904
periods 29-32: 41,784     period     43:  2,112
periods 33-34: 54,312     periods 44-68:     24
periods 35-41: 53,664 -> 38,112 (declining)
```

It was configured as `min_period_batch_segments=[44]`, `batch_mode="largest_block"`,
and the setup log read:

```
segment 0: periods 44-68 -> 25 batches x 24     (start search: 24,   end search: 24)
segment 1: periods  0-43 -> 747 batches x 2112  (start search: 2112, end search: 2112)
2 segment(s), 773 scan steps total
```

`start search == end search` means the search never shrank: 2,112 is simply period
43's size, and one 2,112-wide period was setting the width for 44 periods of
33,084-54,312. Step floor here is `71 - 2 = 69`. The sweep:

| separators, mode | steps | slots |
| --- | --- | --- |
| `[44]`, `largest_block` (before) | 773 | 1,578,264 |
| `[29, 35, 42, 44]`, `largest_block` | 84 | 1,541,014 |
| `[44]`, `[largest_block, period_max]` | 69 | 2,390,328 |
| `[29, 33, 44]`, `period_max` | 69 | 1,724,604 |
| `[29, 33, 42, 44]`, `period_max` | **69** | **1,657,788** |
| `[29, 33, 35, 42, 44]`, `period_max` | 69 | 1,653,252 |

`[29, 33, 42, 44]` with `period_max` was chosen: the floor, at 5% padded work, from
11x as many steps. Splitting the declining run as well (adding 35) saved a further
0.3% and was not worth a sixth segment.
