# Wave ARC final decision gate

Date: 2026-09-15

## Question

Does the small complex wave/interference architecture provide an ARC generalization
advantage that cannot be explained by an ordinary real local recurrent model with
the same quadratic / square-law relational readout?

This is the final bounded continuation gate for the original neuron/antenna research
direction.  No task-specific tuning was allowed.

## Models

### Simplest surviving wave model

The strongest causal ablation from the earlier frozen-30 experiment was the
`linear` wave model:

- fixed complex 5-neighbor propagation;
- trainable complex local mode mixing shared at every grid cell;
- four recurrent propagation steps;
- persistent input injection;
- no recurrent phase nonlinearity;
- no recurrent amplitude saturation;
- final complex projection followed by square-law power `|E|^2`.

The recurrent hidden dynamics are therefore complex-linear; the detector is
quadratic.

### Real quadratic control

`RealQuadraticARC` was designed as the decisive conventional control:

- fixed real 5-neighbor propagation;
- trainable real local mode mixing shared at every grid cell;
- four recurrent linear steps;
- persistent input injection;
- two real detector projections per output class;
- squared-sum detector `q_a^2 + q_b^2`.

It has 451 trainable real parameters, close to the 456 parameters allocated by
the full wave model.  The readout has the same polynomial order and two detector
quadratures, but the hidden state has no complex phase/interference constraint.

## Task sets

The ARC-AGI-2 training corpus was filtered exactly as in the original kill test:

- shape-preserving tasks only;
- maximum grid size 400 cells;
- deterministic shuffle seed `20260915`.

Two disjoint 30-task slices were used:

1. **Development/frozen set:** shuffled positions 1-30.  This is the set used by
   the earlier architecture experiments.
2. **Fresh validation set:** shuffled positions 31-60.  These tasks had zero
   overlap with the development set and were not used to choose the architecture.

Each model was trained independently for 500 Adam updates at learning rate 0.015
for seeds 0, 1, and 2.  ARC exact match and pixel accuracy were recorded.

## Earlier controls closed before the final gate

On the original frozen 30 tasks:

| Model | Exact task-seed runs | Mean test pixel | Robust 3/3 tasks |
|---|---:|---:|---:|
| full complex wave | 12/90 | ~76.2% | 3 |
| wave without learned phase nonlinearity | 11/90 | 76.29% | 3 |
| complex-linear wave + square-law detector | 10/90 | 76.57% | 2 |
| real recurrent, 13 channels / 442 params | 5/90 | ~73.0% | 0 |
| real recurrent, 16 channels / 592 params | 7/90 | 73.63% | 1 |

Therefore neither the learned intensity-dependent phase nonlinearity nor the
number of real hidden coordinates explains the earlier result.  The remaining
candidate was the complex/interference representation and its quadratic detector.

## Final gate results

### Development/frozen 30 tasks

| Model | Exact task-seed runs | Robust 3/3 tasks | Majority >=2/3 tasks | Mean test pixel | Fully fit training runs |
|---|---:|---:|---:|---:|---:|
| complex-linear wave | 10/90 | 2 | 4 | 76.53% | 28/90 |
| real quadratic | **13/90** | **3** | **5** | 76.27% | 31/90 |

The real quadratic model matches or exceeds the complex-wave model on exact ARC
solves.  Among runs that exactly fit all training demonstrations, exact test
transfer was 35.7% for complex wave and 41.9% for real quadratic.

### Fresh untouched 30 tasks

| Model | Exact task-seed runs | Robust 3/3 tasks | Majority >=2/3 tasks | Mean test pixel | Fully fit training runs |
|---|---:|---:|---:|---:|---:|
| complex-linear wave | **2/90** | 0 | 0 | **76.94%** | 37/90 |
| real quadratic | **2/90** | 0 | 0 | 75.74% | 34/90 |

The exact-match advantage disappears completely on fresh tasks.

The complex model retains a small average pixel advantage of approximately
+1.20 percentage points.  Task-level bootstrap over the 30 fresh tasks gives a
95% interval of approximately +0.33 to +2.15 percentage points.  This is a real
small local-prediction effect in this sample, but it does not produce more exact
ARC solutions.

Among runs that exactly fit all demonstrations, fresh-set exact transfer was:

- complex-linear wave: 2 / 37 = 5.4%;
- real quadratic: 2 / 34 = 5.9%.

Thus the models are essentially equally poor at the abstraction criterion that
matters for ARC.

### Combined 60-task picture

Across both disjoint sets (180 task-seed runs per model):

- complex-linear wave exact solves: 12/180;
- real quadratic exact solves: 15/180;
- paired full-exact outcomes: wave-only 3, real-only 6, both 9;
- mean pixel advantage for wave: approximately +0.73 percentage points.

The small pixel-level advantage is not evidence for a general reasoning
advantage.

## Interpretation

The experiments do **not** support the claim that complex wave/interference
physics provides a unique ARC abstraction mechanism in this architecture.

The earlier successes are parsimoniously explained by a much simpler mechanism:

1. repeated shared local propagation creates a bounded spatial receptive field;
2. linear recurrent mixing creates structured local features;
3. square-law / quadratic readout exposes pairwise spatial relations;
4. some ARC tasks happen to align strongly with this local quadratic hypothesis
   class.

The complex representation changes the detailed bias and gives a small mean
pixel-accuracy improvement, but an ordinary real quadratic learner can reproduce
or exceed the exact task-solving behavior.

## Decision

**Stop the wave/antenna architecture as a primary ARC-AGI / general-intelligence
research direction.**

Do not spend additional time scaling this model, adding more wave nonlinearities,
or attempting a large ARC solver based on it unless a new independent reason
emerges.

The reusable scientific insight is narrower:

> ARC contains task families where a strongly constrained local quadratic
> relational hypothesis class generalizes from very few demonstrations better
> than a more flexible memorizing baseline.

That observation may be useful for representation design, kernels, cellular
models, or symbolic-neural hybrids, but it is not currently evidence that the
physical-wave substrate itself is the important ingredient.

This is a successful negative result: the bounded kill test did what it was
supposed to do and prevented further investment in a hypothesis that did not
survive its strongest conventional control.
