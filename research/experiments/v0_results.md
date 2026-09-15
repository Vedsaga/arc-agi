# v0 results: observational quotient baseline

Source artifacts: `semantic_quotient_eval.csv`, `semantic_quotient_test.csv`, and `submission.json` from the first Kaggle run of `semantic_quotient_search_arc_agi2_v0.ipynb`.

## Public evaluation set

- Tasks: **120**
- Test outputs scored: **172**
- Exact outputs solved by either attempt: **0 / 172**
- Raw programs generated per task: **72**
- Mean observational behavior classes per task: **13.67**
- Median observational behavior classes per task: **15**
- Mean raw-program / behavior-class ratio: **5.59x**
- Median ratio: **4.8x**
- Maximum ratio: **9x**
- Exact demonstration-fitting survivors: **0 on every evaluation task**
- Mean runtime per task: about **0.021 s**

## Visible competition test challenges

- Tasks: **240**
- Median raw-program / behavior-class ratio: **4.8x**
- Three visible tasks had one exact demonstration-fitting candidate under the v0 DSL: `0d3d703e`, `1cf80156`, `3c9b0459`.
- Hidden correctness cannot be inferred from this file because test solutions are not available.

## What this means

The first experiment found substantial **syntactic-to-behavioral redundancy**: many different generated programs behaved identically on the demonstrations. However, the DSL had essentially no coverage of the ARC-AGI-2 evaluation tasks: no program exactly matched all demonstrations on any public evaluation task.

Therefore v0 does **not** establish that quotienting improves ARC accuracy. It establishes only that redundancy is measurable in this small hypothesis language.

There are also two conceptual issues to fix in v1:

1. v0 quotienting happened **after** program generation, so it measured redundancy but did not save generation/search work.
2. Quotienting only by behavior on the demonstrations is too aggressive. Any two perfectly demonstration-consistent programs necessarily agree on those demonstrations, yet they may generalize differently to the unseen test input. v1 should use richer counterfactual/probe signatures and/or retain multiple representatives per class.

The v0 `hartley` column should not be interpreted as posterior uncertainty: it counted all quotient classes, including classes inconsistent with the target outputs. v1 will rename/redefine this metric.

## v1 targets

Measure separately:

- **Coverage:** fraction of tasks with at least one program that exactly fits every demonstration.
- **Search reduction:** ratio between the syntactic search tree size and the number of candidates actually expanded after quotient pruning.
- **Safety/generalization:** whether probe-based quotienting preserves or improves validation accuracy relative to less aggressive pruning.
- **Ambiguity:** number of distinct test predictions produced by demonstration-consistent hypotheses.
