"""Neighbor-list accelerated inter-body Debye-Huckel repulsion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from FMBD.backend import to_numpy
from FMBD.core.forces import NodalWrench
from FMBD.interaction.base import InteractionResult


def mgcl2_ionic_strength_mM(concentration_mM: float) -> float:
    return 3.0 * concentration_mM


@dataclass
class DebyeHuckelInteraction:
    body_a: str
    body_b: str
    charge: float | Callable[[Any], float] = 0.7
    concentration_key: str = "Mg_mM"
    ionic_strength: Callable[[float], float] = mgcl2_ionic_strength_mM
    temperature: float = 300.0
    relative_permittivity: float = 80.0
    cutoff_multiplier: float = 4.0
    skin: float = 2.0
    initial_pair_cutoff: float | None = None
    freeze_neighbor_list: bool = False
    apply_force_cutoff: bool = True
    rebuild_check_every: int = 1000
    min_distance: float = 0.2
    search_chunk_a: int = 256
    search_chunk_b: int = 1024
    force_chunk_size: int = 100000
    excluded_nodes_a: Sequence[int] = field(default_factory=tuple)
    excluded_nodes_b: Sequence[int] = field(default_factory=tuple)
    name: str = "electrostatic_repulsion"
    _pairs: Any = field(default=None, init=False, repr=False)
    _reference_a: Any = field(default=None, init=False, repr=False)
    _reference_b: Any = field(default=None, init=False, repr=False)
    _list_cutoff: float | None = field(default=None, init=False, repr=False)
    _rebuild_count: int = field(default=0, init=False, repr=False)
    _gpu_pair_search: Any = field(default=None, init=False, repr=False)

    def initialize(self, system: Any, context: Any) -> None:
        if self.body_a not in system.bodies or self.body_b not in system.bodies:
            raise KeyError("Debye-Huckel body names must exist in the system")
        for name, indices in ((self.body_a, self.excluded_nodes_a), (self.body_b, self.excluded_nodes_b)):
            n = system.bodies[name].model.n_node
            if any(i < 0 or i >= n for i in indices):
                raise IndexError(f"Debye-Huckel exclusion index is out of range for body {name!r}")
        if system.bodies[self.body_a].model.backend != system.bodies[self.body_b].model.backend:
            raise ValueError("interacting bodies must use the same array backend")
        if system.bodies[self.body_a].model.backend == "cuda":
            from FMBD.interaction.gpu_neighbors import GPUCrossBodyPairSearch
            device = int(system.bodies[self.body_a].model.X.device.id)
            self._gpu_pair_search = GPUCrossBodyPairSearch(device=device)

    def _parameters(self, context: Any):
        concentration = float(context.environment[self.concentration_key])
        ionic = self.ionic_strength(concentration) * 1.0e-27
        if ionic <= 0.0:
            raise ValueError("Debye-Huckel ionic strength must be positive")
        eps0 = 8.854187817e-12 * (6.24150934e18) ** 2 * 1.0e-30
        kBT = 1.3806504e-23 * 1.0e21 * self.temperature
        ld = math.sqrt((eps0 * self.relative_permittivity * kBT) / (2.0 * 6.02214076e23 * ionic))
        charge = float(self.charge(context) if callable(self.charge) else self.charge)
        return eps0, ld, 1.0 / ld, charge

    def _build_pairs(self, xa, xb, cutoff):
        xp = self._xp
        if xp is not np:
            if self._gpu_pair_search is None:
                from FMBD.interaction.gpu_neighbors import GPUCrossBodyPairSearch
                self._gpu_pair_search = GPUCrossBodyPairSearch(device=int(xa.device.id))
            return self._gpu_pair_search.query(
                xa, xb, cutoff,
                excluded_a=self.excluded_nodes_a,
                excluded_b=self.excluded_nodes_b,
            )
        active_a = np.ones(xa.shape[0], dtype=bool)
        active_b = np.ones(xb.shape[0], dtype=bool)
        active_a[np.asarray(self.excluded_nodes_a, dtype=np.int64)] = False
        active_b[np.asarray(self.excluded_nodes_b, dtype=np.int64)] = False
        ids_a, ids_b = np.flatnonzero(active_a), np.flatnonzero(active_b)
        tree = cKDTree(xb[ids_b])
        pairs = [(ids_a[i], ids_b[j]) for i, found in enumerate(tree.query_ball_point(xa[ids_a], cutoff)) for j in found]
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        body_a, body_b = system.bodies[self.body_a], system.bodies[self.body_b]
        if body_a.reconstruction is None or body_b.reconstruction is None:
            raise RuntimeError("interacting bodies must be reconstructed")
        xa, xb = body_a.reconstruction.x, body_b.reconstruction.x
        xp = body_a.model.xp
        self._xp = xp
        eps0, ld, kappa, charge = self._parameters(context)
        force_cutoff = self.cutoff_multiplier * ld
        list_cutoff = self.initial_pair_cutoff if self.initial_pair_cutoff is not None else force_cutoff + self.skin
        rebuild = self._pairs is None or self._list_cutoff is None or abs(self._list_cutoff - list_cutoff) > 1.0e-12
        if not rebuild and not self.freeze_neighbor_list and context.step % self.rebuild_check_every == 0:
            displacement = xp.max(xp.linalg.norm(xa - self._reference_a, axis=1)) + xp.max(xp.linalg.norm(xb - self._reference_b, axis=1))
            rebuild = float(to_numpy(displacement)) > self.skin
        if rebuild:
            self._pairs = self._build_pairs(xa, xb, list_cutoff)
            self._reference_a, self._reference_b = xa.copy(), xb.copy()
            self._list_cutoff = list_cutoff
            self._rebuild_count += 1
        wa = NodalWrench.zeros(body_a.model.n_node, xp=xp, dtype=xa.dtype)
        wb = NodalWrench.zeros(body_b.model.n_node, xp=xp, dtype=xb.dtype)
        energy = xp.asarray(0.0, dtype=xa.dtype) if diagnostics else None
        n_used, minimum = 0, math.inf
        prefactor = charge**2 / (4.0 * math.pi * eps0 * self.relative_permittivity)
        for start in range(0, len(self._pairs), self.force_chunk_size):
            pair = self._pairs[start:start + self.force_chunk_size]
            ia, ib = pair[:, 0], pair[:, 1]
            rvec = xa[ia] - xb[ib]
            raw = xp.linalg.norm(rvec, axis=1)
            active = raw <= force_cutoff if self.apply_force_cutoff else xp.ones_like(raw, dtype=bool)
            r = xp.maximum(raw, self.min_distance)
            exponential = xp.exp(-kappa * r)
            active_f = active.astype(r.dtype)
            magnitude = prefactor * exponential * (kappa / r + 1.0 / r**2) * active_f
            force = magnitude[:, None] * rvec / xp.maximum(raw, 1.0e-12)[:, None]
            xp.add.at(wa.force, ia, force)
            xp.add.at(wb.force, ib, -force)
            if diagnostics:
                energy += xp.sum(prefactor * exponential / r * active_f)
                n_used += int(to_numpy(active.sum()))
                if bool(to_numpy(active.any())):
                    minimum = min(minimum, float(to_numpy(xp.min(xp.where(active, raw, xp.inf)))))
        detail = {}
        if diagnostics:
            detail = {
                "n_candidate": int(len(self._pairs)), "n_used": n_used,
                "min_distance": minimum, "neighbor_rebuild_count": self._rebuild_count,
                "debye_length": ld, "force_cutoff": force_cutoff if self.apply_force_cutoff else math.inf,
                "initial_pair_cutoff": list_cutoff, "neighbor_list_frozen": self.freeze_neighbor_list,
                "effective_charge": charge, "body_a_net_force_pN": wa.force.sum(axis=0),
                "body_a_torque_about_center_pN_nm": xp.cross(body_a.reconstruction.xrel, wa.force, axis=1).sum(axis=0),
            }
        return InteractionResult(nodal_wrenches={self.body_a: wa, self.body_b: wb}, energy=energy, diagnostics=detail)
