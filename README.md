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

From the repository root, install with `python -m pip install -e .`.
PDB/DCD writing uses NumPy and does not require MDAnalysis. The optional
`.[trajectory]` extra installs MDAnalysis for reading and inspecting outputs.

### Safe live trajectories

`DCDTrajectoryWriter(path, every=100, publish_every=10)` saves coordinates
every 100 integration steps and publishes a complete viewer-readable DCD
every 10 saved frames. Initial and final publication are always attempted.
The append-only `.<filename>.stream` journal is flushed and closed after each
frame; `path` is replaced atomically from a temporary snapshot. A Windows
viewer locking `path` only delays publication, never integration or journal
recording. Reopen the public DCD to see its latest snapshot. If it is still
locked at completion, a `<stem>_complete.dcd` final copy is published instead.
The journal retains all recorded frames for recovery.

Snapshots are copied with bounded memory, but copy cost grows with trajectory
length. Increase `publish_every` for long runs; it does not change sampling or
discard any saved frames. Both topology-PDB and DCD coordinates use Angstrom.
Pass `InterBodyBond` entries to `PDBTopologyWriter` to include linker CONECT
bonds; an implicit ssDNA linker is a bond, not an explicit nucleotide chain.

### Single-strand connectors

`FMBD.interaction.SingleStrandBeamInteraction` uses the PySNUPI mean ssDNA
material law, including length-dependent axial rigidity and corotational
bending/torsion. `SingleStrandDNAParameters.from_nucleotides(16)` means 16
unpaired nucleotides plus two attachment bases (bridge count 18). It determines
the axial rest length from the material rather than initial body placement.
Angular reference frames are captured at initialization. Nodal force/moment
are evaluated analytically in batches; potential quadrature runs only for
requested diagnostics. The midpoint integrator and generalized-force
projection are unchanged.

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
