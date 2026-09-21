# ai_instructions

Procedures written for an AI agent to follow on a *specific* dcegm model, where the
right answer depends on that model's numbers rather than on the library.

These are not documentation of how dcegm works -- that lives in `docs/`, and each
procedure links to the relevant page. They are task recipes: what to measure, how to
measure it, how to decide, and what to check before believing the result.

| file                                   | task                                                                                                                 |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [`tune_batching.md`](tune_batching.md) | Choose `min_period_batch_segments` and `batch_mode` for a model, to cut the number of backward-induction scan steps. |

## Conventions for anything added here

- **Measure, don't reason from the shape of the code.** Each procedure should say which
  script to run and what its output looks like. A recommendation with no numbers behind
  it is not finished.
- **State the floor.** Say what the best achievable result is and why, so the reader
  knows when to stop optimising.
- **Report memory as a ratio.** Absolute bytes need the target device and do not
  transfer; the ratio between the candidates does, and it is what decides whether a
  configuration fits. Name the quantity the ratio is taken over.
- **Carry the pitfalls.** The gotchas that cost an hour the first time (a stripped model
  pickle, a numpy/jax dispatch error) belong in the file, not in a commit message.
- **Separate the library-general part from the worked example.** The procedure must
  transfer to a different model; the example is there to show what the output looks
  like, not to be copied.
