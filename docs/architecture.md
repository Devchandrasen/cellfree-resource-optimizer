# Architecture

The public API separates physical network construction, optimization, independent validation and output files. No solver reads an earlier run directory.

```mermaid
flowchart LR
  C[NetworkConfig] --> G[Statistical channel generator]
  G --> N[Network moments and positions]
  N --> B[Fresh method and runtime budget]
  B --> S[Association and serving-mode search]
  S --> P[Conic and linear power / payload solves]
  P --> V[Physical candidate checks]
  V -->|accepted before deadline| I[Retained incumbent]
  I --> R[Independent exported-point validation]
  R --> E[NPZ / JSON / MAT / plots]
```

## Package layout

```text
src/cellfree_optimizer/
├── config.py          Validated network settings
├── network.py         Geometry and finite-MMSE moments
├── optimizer.py       Public method dispatch and acceptance contract
├── validation.py      Non-repairing physical constraint checks
├── io.py              Portable files and environment records
├── cli.py             Runs, batches, reports and validation
├── visualization.py   Optional run overviews
└── _engine/           Internal numerical implementation
```

The engine contains coupled searches, coherent initialization, AP-capacity transitions, mode probes, structured conic kernels, payload recovery and conservative numerical bound helpers. `_engine` is an implementation detail. Use the package API for normal execution.

## Allocation arrays

| Field | Shape | Meaning |
| :--- | :--- | :--- |
| `A` | APs × users | Binary selected serving links |
| `z` | users | True for coherent service, false for noncoherent service |
| `x` | APs × users | Nonnegative square root of the normalised AP power share |
| `rC` | users | Coherent payload rates |
| `u` | APs × users | Noncoherent payload carried by each AP stream |
| `t` | scalar | Minimum delivered user service |

Each AP has a unit normalised power budget. Its physical transmit budget in watts enters the channel coefficients. The validation layer checks power, support, binary variables, mode consistency, radio ceilings, fronthaul load, coverage and association limits.

Python arrays use zero-based indexing. `order` and `rank` are converted to one-based indexing only for the MATLAB export.
