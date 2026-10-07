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

The supported runtime uses Torch tensors on CPU and CUDA. It provides a stable
body-data schema, geometry and force abstractions, stateful interactions,
Brownian integration, live logging, and PDB/DCD observers. The default
integrator is stochastic midpoint Brownian; the former Euler Brownian update
remains selectable through `IntegratorOptions(scheme="euler_brownian")`.

From the repository root, install with
`python -m pip install -e ".[trajectory]"` when PDB/DCD trajectory observers
are required.

For CUDA execution, install the Torch build matching the server driver/runtime,
then install FMBD-SNUPI normally. A model selects a device when it is loaded:
`BodyModel.from_fmbd_data(path, device="cuda:0")`.

The intermediate NumPy/SciPy and CuPy implementation is retained under
`FMBD.legacy_numpy_cupy` for result reproduction and future backend work. It is
not the default runtime. Install `.[legacy-numpy-cupy]`,
`.[legacy-cuda12]`, or `.[legacy-cuda13]` only when that archived backend is
required. The old `cuda`, `cuda12`, and `cuda13` extras remain compatibility
aliases for environments created with FMBD-SNUPI 0.1.

PySNUPI's `snupy.mor_fmbd_input` pickle is loaded directly by
`BodyModel.from_fmbd_data(...)`; FMBD does not import PySNUPI at runtime.

See [the Korean quick-start guide](docs/quickstart_ko.md) and
[the two-body baseline template](examples/baseline_two_body.py).
