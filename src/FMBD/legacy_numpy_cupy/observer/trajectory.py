"""Generic PDB topology and DCD coordinate trajectory observers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np

from FMBD.legacy_numpy_cupy.backend import to_numpy


@dataclass(frozen=True)
class InterBodyBond:
    """A topology bond between nodes belonging to two named bodies."""

    body_a: str
    node_a: int
    body_b: str
    node_b: int


def _body_offsets(system: Any) -> dict[str, int]:
    offset, offsets = 0, {}
    for name, body in system.bodies.items():
        offsets[name] = offset
        offset += body.model.n_node
    return offsets


def _combined_bonds(system: Any, inter_body_bonds: Sequence[InterBodyBond]) -> np.ndarray:
    offsets = _body_offsets(system)
    blocks: list[np.ndarray] = []
    for name, body in system.bodies.items():
        if len(body.model.connectivity):
            blocks.append(body.model.connectivity + offsets[name])
    for bond in inter_body_bonds:
        if bond.body_a not in system.bodies or bond.body_b not in system.bodies:
            raise KeyError("inter-body bond names must be registered bodies")
        a, b = system.bodies[bond.body_a], system.bodies[bond.body_b]
        if not 0 <= bond.node_a < a.model.n_node or not 0 <= bond.node_b < b.model.n_node:
            raise IndexError("inter-body bond node index out of range")
        blocks.append(np.array([[offsets[bond.body_a] + bond.node_a, offsets[bond.body_b] + bond.node_b]], dtype=np.int64))
    if not blocks:
        return np.empty((0, 2), dtype=np.int64)
    bonds = np.vstack(blocks)
    bonds = bonds[bonds[:, 0] != bonds[:, 1]]
    return np.unique(np.sort(bonds, axis=1), axis=0)


def _write_conect_records(handle, bonds: np.ndarray) -> None:
    adjacency: dict[int, set[int]] = {}
    for i0, j0 in bonds:
        i, j = int(i0) + 1, int(j0) + 1
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    for atom_id, neighbors in sorted(adjacency.items()):
        ordered = sorted(neighbors)
        for start in range(0, len(ordered), 4):
            handle.write(f"CONECT{atom_id:5d}" + "".join(f"{node:5d}" for node in ordered[start:start + 4]) + "\n")


def _format_pdb_coordinate(value: float) -> str:
    """Format one PDB Real(8.n) coordinate without spilling into a neighbor.

    Standard PDB permits only eight characters per coordinate.  Prefer 0.001 Å
    precision, then reduce decimal places only when a large magnitude requires
    it.  For example, ``-1412.858`` becomes ``-1412.86``.
    """
    for precision in range(3, -1, -1):
        formatted = f"{value:8.{precision}f}"
        if len(formatted) == 8:
            return formatted
    raise ValueError(f"PDB coordinate cannot fit into an 8-character field: {value}")


def _snupy_segment_name(helix_id: int) -> str:
    """Match SNUPY's PDB SEGID convention without repurposing it for bodies."""
    helix_id = int(helix_id)
    return f"H{helix_id}" if helix_id < 1000 else f"C{helix_id - 1000}"


def _snupy_topology(body: Any) -> dict[str, Any] | None:
    """Return validated SNUPY node topology embedded in an FMBD artifact.

    The artifact is deliberately optional so legacy models retain the generic
    FMBD PDB writer behavior.
    """
    topology = body.model.metadata.get("snupy_topology")
    if topology is None:
        return None
    required = ("seq_node", "helix_id", "resid", "etbr_flag", "dox_flag", "chain_id")
    missing = [key for key in required if key not in topology]
    if missing:
        raise KeyError(f"SNUPY topology is missing fields: {missing}")
    n_node = body.model.n_node
    arrays = {key: np.asarray(topology[key]) for key in required[:-1]}
    invalid = [key for key, value in arrays.items() if value.shape != (n_node,)]
    if invalid:
        raise ValueError(f"SNUPY topology fields must have one entry per node: {invalid}")
    chain_id = str(topology["chain_id"])
    if len(chain_id) != 1 or not chain_id.isalnum():
        raise ValueError("SNUPY topology chain_id must be one alphanumeric PDB character")
    return {**arrays, "chain_id": chain_id, "atom_name": str(topology.get("atom_name", "N")), "element": str(topology.get("element", "N"))}


@dataclass
class PDBTopologyWriter:
    """Write one PDB topology from the initial reconstructed geometry."""

    path: str | Path
    inter_body_bonds: Sequence[InterBodyBond] = field(default_factory=tuple)
    coordinate_scale: float = 10.0  # nm -> Å

    def on_start(self, simulation: Any, context: Any) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bonds = _combined_bonds(simulation.system, self.inter_body_bonds)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("REMARK   FMBD topology; coordinates are the initial reconstructed state\n")
            handle.write("REMARK 950 FMBD BODY CHAIN MAP\n")
            for chain_index, (name, body) in enumerate(simulation.system.bodies.items()):
                topology = _snupy_topology(body)
                chain = topology["chain_id"] if topology is not None else chr(ord("A") + chain_index % 26)
                handle.write(f"REMARK 950 FMBD BODY CHAIN {chain:1s} {name}\n")
            serial = 1
            for chain_index, (name, body) in enumerate(simulation.system.bodies.items()):
                if body.reconstruction is None:
                    raise RuntimeError("system must be reconstructed before PDB output")
                topology = _snupy_topology(body)
                chain = topology["chain_id"] if topology is not None else chr(ord("A") + chain_index % 26)
                resname = name[:3].upper().ljust(3, "X")
                xyz = to_numpy(body.reconstruction.x) * self.coordinate_scale
                for index, (x, y, z) in enumerate(xyz):
                    x_field = _format_pdb_coordinate(float(x))
                    y_field = _format_pdb_coordinate(float(y))
                    z_field = _format_pdb_coordinate(float(z))
                    if topology is None:
                        record_atom, record_resname = "N", resname
                        resid, segid, element = (index % 9999) + 1, "", "N"
                    else:
                        record_atom = topology["atom_name"]
                        record_resname = "ET" if bool(topology["etbr_flag"][index]) else ("DO" if bool(topology["dox_flag"][index]) else str(topology["seq_node"][index]))
                        resid = int(topology["resid"][index])
                        segid = _snupy_segment_name(int(topology["helix_id"][index]))
                        element = topology["element"]
                    handle.write(
                        f"ATOM  {serial:5d} {record_atom:>4s} {record_resname[:3]:>3s} {chain}{resid:4d}    "
                        f"{x_field}{y_field}{z_field}{1.00:6.2f}{0.00:6.2f}      {segid:<4s}{element:>2s}\n"
                    )
                    serial += 1
                handle.write("TER\n")
            _write_conect_records(handle, bonds)
            handle.write("END\n")

    def on_step(self, simulation: Any, result: Any) -> None:
        return None

    def on_finish(self, simulation: Any) -> None:
        return None


@dataclass
class DCDTrajectoryWriter:
    """Write all bodies' reconstructed coordinates to a DCD trajectory.

    MDAnalysis is imported lazily, so users who do not request DCD output do
    not need it installed. Coordinates are converted from nm to Å at output.
    """

    path: str | Path
    every: int = 1
    _writer: Any = field(default=None, init=False, repr=False)
    _universe: Any = field(default=None, init=False, repr=False)
    _last_written_step: int | None = field(default=None, init=False, repr=False)

    def on_start(self, simulation: Any, context: Any) -> None:
        if self.every <= 0:
            raise ValueError("DCD output interval must be positive")
        try:
            import MDAnalysis as mda
            from MDAnalysis.coordinates.DCD import DCDWriter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("DCDTrajectoryWriter requires the optional dependency MDAnalysis") from exc
        n_atoms = sum(body.model.n_node for body in simulation.system.bodies.values())
        self._universe = mda.Universe.empty(n_atoms, n_residues=n_atoms, atom_resindex=np.arange(n_atoms), trajectory=True)
        self._universe.add_TopologyAttr("names", ["N"] * n_atoms)
        self._universe.add_TopologyAttr("types", ["N"] * n_atoms)
        self._universe.add_TopologyAttr("resnames", ["NOD"] * n_atoms)
        self._universe.add_TopologyAttr("resids", np.arange(1, n_atoms + 1))
        self._universe.add_TopologyAttr("segids", ["SYS"])
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = DCDWriter(str(path), n_atoms=n_atoms)
        self._write(simulation, step=0)

    def _write(self, simulation: Any, step: int) -> None:
        if self._writer is None or self._universe is None:
            raise RuntimeError("DCD writer has not been initialized")
        xyz = np.concatenate([to_numpy(body.reconstruction.x) for body in simulation.system.bodies.values()], axis=0)
        self._universe.atoms.positions = xyz * 10.0
        self._universe.dimensions = np.array([0.0, 0.0, 0.0, 90.0, 90.0, 90.0], dtype=np.float32)
        # FMBD systems are non-periodic unless an application provides box
        # dimensions. MDAnalysis represents a zero-sized unit cell as None and
        # warns while writing the corresponding all-zero DCD cell; suppress only
        # that expected output warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No dimensions set for current frame, zeroed unitcell will be written",
                category=UserWarning,
                module=r"MDAnalysis\.coordinates\.DCD",
            )
            self._writer.write(self._universe.atoms)
        self._last_written_step = step

    def on_step(self, simulation: Any, result: Any) -> None:
        completed_step = result.context.step + 1
        if completed_step % self.every == 0:
            self._write(simulation, completed_step)

    def on_finish(self, simulation: Any) -> None:
        if self._writer is None:
            return
        if self._last_written_step != getattr(simulation, "last_completed_step", None):
            self._write(simulation, getattr(simulation, "last_completed_step", 0))
        self._writer.close()
        self._writer = None
        self._universe = None
