# FMBD-SNUPI

`FMBD-SNUPI` is an independent runtime package for flexible multibody Brownian
dynamics. PySNUPI remains an offline preprocessor that produces reduced-order
body data; this package owns reconstruction, interactions, dynamics, protocols
and output orchestration.

The distribution name is `FMBD-SNUPI`; the Python import namespace remains
`FMBD`:

```python
from FMBD import FMBDSystem
```

Internal units are nm, pN, ps, pN nm and rad.

The package provides a stable body-data schema, geometry and force
abstractions, NumPy/SciPy CPU execution, CuPy CUDA execution, and an isolated
legacy Torch implementation.

From the repository root, install with
`python -m pip install -e ".[trajectory]"` when PDB/DCD trajectory observers
are required.

For Debye-Hückel interactions, CPU runs build cross-body neighbor lists with
SciPy `cKDTree`. CUDA runs use a CuPy cell-list kernel and retain candidate
pairs as a Verlet list on the GPU; only the configured exclusions and active
interaction cutoff are applied during force evaluation.

See [the Korean quick-start guide](docs/quickstart_ko.md) and
[the two-body baseline template](examples/baseline_two_body.py).
