# Wave ARC Kill Test

Date: 2026-09-15

## Question

Does the surviving neuron/antenna idea contribute anything useful to ARC-style abstraction, or is it only a different substrate for ordinary pattern fitting?

This experiment is intentionally a **kill test**, not a benchmark submission and not a claim that a wave model is a general ARC solver.

The first gate restricts to ARC-AGI-2 training tasks where every input and output grid has the same shape. That removes output-size inference so the experiment can isolate few-shot spatial transformation learning.

## Candidate architecture

Each grid cell is treated as a fixed physical site with a small complex state (`C` modes).

At every recurrent step:

1. fixed complex coupling propagates state to neighboring sites;
2. a trainable complex mode mixer is applied locally and shared over all sites;
3. an intensity-dependent phase response and saturating amplitude provide nonlinearity;
4. the original input excitation remains weakly injected;
5. output colors are read by square-law detector power.

The model therefore has no free weight per pixel and no ARC-specific transformation DSL.

## Controls

The script compares against:

- `cell_color_map`: a trivial same-location color mapping;
- `conv`: a small conventional recurrent CNN with coordinate channels and comparable state width/depth.

The current repository's symbolic v2 experiment is another contextual control, but it is **not** mixed into this experiment. It previously obtained 0/50 exact held-out tasks because its DSL lacked representation coverage.

## First manual probes

### `a699fb00` — local relation

Rule: fill the zero between horizontally separated color-1 cells with color 2.

Small neighbor-wave model, one seed:

- train exact: 100%
- test exact: 100%
- test pixel accuracy: 100%

This proves only that the architecture can learn a reusable local spatial relation from three demonstrations.

### `a85d4709` — factorization/global broadcast

Rule: in each row, the column containing color 5 determines the output color for the whole row.

Neighbor-wave model:

- train exact: 100%
- held-out exact: 0%
- held-out pixel accuracy in the initial run: 77.8%

Adding generic positional waves did not reliably fix the task.

A dense Green-function-like global wave propagator also fit all demonstrations but failed the held-out grid across five seeds (~22–33% pixel accuracy).

A conventional recurrent CNN with coordinate channels likewise fit the training demonstrations but failed the held-out grid.

### Interpretation

The failure is not simply lack of propagation range. The task becomes trivial after the representation is factorized into independent rows, but neither generic differentiable learner discovers that factorization from four grids. This is exactly the abstraction problem ARC is designed to expose.

## Pre-registered decision rule

Run a deterministic sample of 30 shape-preserving ARC training tasks, using their hidden-within-file `test` outputs only for scoring.

Continue the direct wave->ARC path only if all of the following hold:

1. The wave learner obtains **non-zero exact task success** across the sample.
2. It solves at least one exact task that is not solved by the trivial color-map baseline.
3. At least one wave-only success is not also obtained by the matched recurrent CNN under the same training budget, **or** the wave model reaches the same exact solution with a materially smaller state/parameter/compute budget.
4. Successful tasks replicate across seeds rather than relying on one initialization.
5. Success is not explainable by test leakage, per-pixel free parameters, or a hand-written ARC transformation rule.

If these conditions fail, stop treating the current wave architecture as a promising direct ARC solver. The physical-wave work may still be useful as a representation-learning research program, but it should not consume time aimed at ARC-AGI.

## Run locally

From the repository root:

```bash
python research/experiments/wave_arc_kill_test.py \
  --limit 30 \
  --seeds 0,1,2 \
  --iterations 500 \
  --output research/experiments/wave_arc_kill_test_results.csv
```

Run the two manual probe tasks only:

```bash
python research/experiments/wave_arc_kill_test.py \
  --task-ids a699fb00,a85d4709 \
  --seeds 0,1,2
```

For a cheap first laptop run, reduce to one seed and 300 updates:

```bash
python research/experiments/wave_arc_kill_test.py --limit 30 --seeds 0 --iterations 300
```

## What not to do yet

Do not add:

- a larger wave model just because a task fails;
- object extraction rules copied from ARC solutions;
- output-shape prediction;
- LLM-generated programs;
- search over a transformation DSL;
- task-specific architectural changes.

Those would prevent this experiment from answering the original question.
