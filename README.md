# FMBD

`FMBD` is an independent runtime package for flexible multibody Brownian
dynamics.  SNUPY remains an offline preprocessor that produces reduced-order
body data; this package owns reconstruction, interactions, dynamics, protocols
and output orchestration.

Internal units are nm, pN, ps, pN nm and rad.

The first migration milestone provides the stable body-data schema and core
geometry/force abstractions.  The reference implementation remains
`../switch_cycle.py` until its deterministic trajectory regression is migrated.

Install `FMBD[trajectory]` when PDB/DCD trajectory observers are required.

For Debye-Hückel interactions, CPU runs build cross-body neighbor lists with
SciPy `cKDTree`. CUDA runs use a CuPy cell-list kernel and retain candidate
pairs as a Verlet list on the GPU; only the configured exclusions and active
interaction cutoff are applied during force evaluation.

See [the Korean quick-start guide](docs/quickstart_ko.md) and
[the two-body baseline template](examples/baseline_two_body.py).
