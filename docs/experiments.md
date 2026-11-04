# Running matched experiments

## Two independent seeds

`NetworkConfig.seed` controls geometry, shadowing and link-capacity assignment. The optimizer seed controls starts and search decisions. Reusing a network seed gives the same physical input. Changing only fronthaul keeps the channel statistics unchanged.

Network generation happens before the acceptance clock. Every method independently constructs its association template after its clock starts. Numerical libraries are limited to one native thread during an optimization call. Processor speed, operating-system scheduling and library versions can still change a timed search trajectory.

## Plan the runtime

Approximate sequential runtime is:

```text
conditions × layouts × optimizer seeds × methods × seconds per method
```

Five layouts, two optimizer seeds and five methods at 60 seconds require roughly 50 minutes, plus generation, solver cleanup and file-writing overhead. The built-in batch runner is sequential so competing methods do not contend for processor time.

```bash
cellfree benchmark --config examples/default.json --methods joint,strong-cjt,pure-ncjt,fixed-hybrid,swap-switch --layouts 5 --seeds 2 --seconds 60 --output runs/comparison
```

The default network seed is increased by one for each layout. Optimizer seeds are `0` through `seeds - 1`. Method order is deterministically shuffled within each matched layout and seed block. Each method receives a fresh independent runtime budget.

## Summaries

`summary.csv` and `summary.json` contain mean delivered service and paired percentage gains relative to `strong-cjt` when that method is present. Ratios are paired at the layout and seed level, then averaged across seeds within each layout. Confidence intervals use layouts as the independent units.

Intervals are pointwise 95% Student intervals. They are not adjusted for multiple comparisons. A single layout does not produce a confidence interval. A zero-service reference cannot form a ratio, and only complete paired seed blocks contribute to paired intervals. The `paired_layouts` column gives that count.

The manifest is updated after every completed run. A batch is marked complete only after all requested runs finish. Reporting rejects an incomplete batch. Interrupted batches remain inspectable, but automatic resume and partial-batch inference are not supplied.

## Output integrity

Every result records SHA256 hashes for its physical inputs, allocation arrays and MATLAB export. It also records package versions, configuration, optimizer seed, accepted-point time, wall time and feasibility residuals. Existing output directories are rejected.

Keep complete batches, signed losses and ties when comparing methods. A finite-budget comparison measures the implemented searches under the selected conditions. It does not identify the globally optimal value.
