# ARC-AGI Semantic Quotient Search

Experimental ARC-AGI-2 research repo exploring a simple hypothesis:

> Search should operate over task-distinguishable behaviors rather than every syntactically distinct program.

## Kaggle notebook

`notebooks/semantic_quotient_search_arc_agi2_v0.ipynb`

The notebook is self-contained and uses only NumPy/pandas plus the ARC Prize 2026 ARC-AGI-2 competition data. In Kaggle:

1. Join **ARC Prize 2026 - ARC-AGI-2**.
2. Create/import a notebook and attach the competition data.
3. Import/open `notebooks/semantic_quotient_search_arc_agi2_v0.ipynb`.
4. Run all cells.
5. Inspect the public evaluation accuracy and `semantic_quotient_eval.csv` diagnostics.
6. The final cell writes `submission.json` plus `semantic_quotient_test.csv`.
7. Submit `submission.json` through the competition workflow.

## What v0 measures

- raw symbolic hypotheses generated per task
- observational equivalence classes on demonstration inputs
- quotient ratio (`raw / classes`)
- exact demonstration-surviving behavior classes
- Hartley uncertainty `log2(number of behavior classes)`
- wall-clock search time
- exact ARC output accuracy using either of the two allowed attempts

Here, **semantic equivalence** is intentionally restricted to *observational equivalence under the task demonstrations*. It is not claimed to be universal semantic equivalence.

## Current DSL

- identity
- rotations
- horizontal/vertical flips
- transpose
- crop to non-background bounding box
- depth-2 compositions
- task-learned global color remappings

This is intentionally a small baseline. The goal of v0 is to make the information/search hypothesis measurable before increasing solver power.

## Planned experiments

1. Incremental quotienting while expanding the program tree, so equivalent branches are pruned before generating descendants.
2. Object/relation primitives: components, containment, alignment, symmetry completion, repetition, counting, line extension, hole filling, and object-to-object mappings.
3. Non-uniform priors over behavior classes and Shannon entropy diagnostics.
4. Information-gain-per-compute scheduling for selecting which hypothesis families to expand.
5. Minimal-demonstration ablations: measure which ARC examples actually eliminate the successful competing rules.
6. Learned proposal mechanisms while retaining exact symbolic verification and behavioral quotienting.

## Research question

> Does searching over task-distinguishable behavior classes reduce the computation required to reach a correct ARC solution compared with searching over syntactically distinct programs?
