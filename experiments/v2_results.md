# v2 Object/Relation Search — Held-out Results

Date: 2026-09-13

## Setup

- 50 tasks from the deterministic held-out split of ARC training data
- Same DSL/depth for plain search and quotient search
- Quotient search uses counterfactual behavioral probes

## Main results

- Tasks evaluated: 50
- Test outputs represented by those tasks: 52
- Exact task coverage, quotient search: 0/50
- Exact task coverage, plain search: 0/50
- Tasks where the best program exactly matched at least one demonstration pair: 3/50
- Median best raw-pixel agreement: ~0.812 (not a reliable ARC success metric because background-heavy grids inflate it)

### Structural search

- Total programs generated, quotient search: 58,279
- Total programs generated, plain search: 70,454
- Structural-generation reduction: ~17.3%
- Median plain/quotient generated-program ratio: ~1.118x

### Actual candidate-input executions

- Quotient search: 864,340
- Plain search: 236,824
- Quotient/plain execution ratio: ~3.65x

Therefore the current counterfactual-probe implementation is a net compute loss: it generates fewer structural programs but spends substantially more executions evaluating probes.

## Interpretation

The dominant failure is representation coverage, not ranking. Neither plain nor quotient search can express a program that exactly fits all demonstrations for any of these 50 tasks.

Representative held-out tasks show missing capabilities such as:

- local/enclosed-region completion or recoloring;
- conditional transformations parameterized by color/object properties;
- learned displacement and correspondence rules;
- transformations over relations between multiple objects rather than whole-object extraction alone.

The v2 result therefore argues against simply increasing composition depth in the current DSL.

## Decision

Do not optimize ranking or add more counterfactual probes yet.

Next experiment should prioritize task-conditioned relational transformation induction and local geometric operators. Counterfactual fingerprinting should be reintroduced only after exact demonstration coverage becomes nonzero, and its compute cost must be measured against the same candidate-execution accounting used by the plain baseline.
