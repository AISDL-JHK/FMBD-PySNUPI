"""Stateful, neighbor-list accelerated inter-body Debye–Hückel repulsion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from FMBD.core.forces import NodalWrench
from FMBD.interaction.base import InteractionResult


def mgcl2_ionic_strength_mM(concentration_mM: float) -> float:
    """Ionic strength of MgCl₂ expressed in mM: ``0.5(4C + 2C) = 3C``."""
    return 3.0 * concentration_mM


@dataclass
class DebyeHuckelInteraction:
    """Repulsion between all eligible cross-body nodes within a screened cutoff.

    ``charge`` may be a constant effective node charge or a function receiving
    the simulation context.  The default screening assumes MgCl₂ and obtains
    concentration from ``context.environment[concentration_key]``.
    """

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
    search_chunk_a: int = 2048
    search_chunk_b: int = 4096
    force_chunk_size: int = 100000
    excluded_nodes_a: Sequence[int] = field(default_factory=tuple)
    excluded_nodes_b: Sequence[int] = field(default_factory=tuple)
    name: str = "electrostatic_repulsion"
    _pairs: torch.Tensor | None = field(default=None, init=False, repr=False)
    _reference_a: torch.Tensor | None = field(default=None, init=False, repr=False)
    _reference_b: torch.Tensor | None = field(default=None, init=False, repr=False)
    _list_cutoff: float | None = field(default=None, init=False, repr=False)
    _rebuild_count: int = field(default=0, init=False, repr=False)

    def initialize(self, system: Any, context: Any) -> None:
        if self.body_a not in system.bodies or self.body_b not in system.bodies:
            raise KeyError("Debye-Huckel body names must exist in the system")
        for name, indices in ((self.body_a, self.excluded_nodes_a), (self.body_b, self.excluded_nodes_b)):
            n_node = system.bodies[name].model.n_node
            if any(index < 0 or index >= n_node for index in indices):
                raise IndexError(f"Debye-Huckel exclusion index is out of range for body {name!r}")

    def _parameters(self, context: Any) -> tuple[float, float, float, float]:
        concentration = float(context.environment[self.concentration_key])
        ionic_strength = self.ionic_strength(concentration) * 1.0e-27  # mM -> nm^-3
        if ionic_strength <= 0.0:
            raise ValueError("Debye-Huckel ionic strength must be positive")
        eps0 = 8.854187817e-12 * (6.24150934e18) ** 2 * 1.0e-30
        kBT = 1.3806504e-23 * 1.0e21 * self.temperature
        debye_length = math.sqrt((eps0 * self.relative_permittivity * kBT) / (2.0 * 6.02214076e23 * ionic_strength))
        effective_charge = float(self.charge(context) if callable(self.charge) else self.charge)
        return eps0, debye_length, 1.0 / debye_length, effective_charge

    def _build_pairs(self, xa: torch.Tensor, xb: torch.Tensor, cutoff: float) -> torch.Tensor:
        active_a = torch.ones(xa.shape[0], device=xa.device, dtype=torch.bool)
        active_b = torch.ones(xb.shape[0], device=xb.device, dtype=torch.bool)
        if len(self.excluded_nodes_a):
            active_a[torch.as_tensor(self.excluded_nodes_a, device=xa.device)] = False
        if len(self.excluded_nodes_b):
            active_b[torch.as_tensor(self.excluded_nodes_b, device=xb.device)] = False
        indices_a, indices_b, chunks = torch.nonzero(active_a).squeeze(1), torch.nonzero(active_b).squeeze(1), []
        for a0 in range(0, indices_a.shape[0], self.search_chunk_a):
            ia = indices_a[a0:a0 + self.search_chunk_a]
            for b0 in range(0, indices_b.shape[0], self.search_chunk_b):
                ib = indices_b[b0:b0 + self.search_chunk_b]
                nearby = torch.nonzero(torch.cdist(xa[ia], xb[ib]) <= cutoff, as_tuple=False)
                if nearby.numel() > 0:
                    chunks.append(torch.column_stack((ia[nearby[:, 0]], ib[nearby[:, 1]])))
        return torch.cat(chunks) if chunks else torch.empty((0, 2), device=xa.device, dtype=torch.long)

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        body_a, body_b = system.bodies[self.body_a], system.bodies[self.body_b]
        if body_a.reconstruction is None or body_b.reconstruction is None:
            raise RuntimeError("interacting bodies must be reconstructed")
        xa, xb = body_a.reconstruction.x, body_b.reconstruction.x
        eps0, ld, kappa, charge = self._parameters(context)
        force_cutoff = self.cutoff_multiplier * ld
        list_cutoff = self.initial_pair_cutoff if self.initial_pair_cutoff is not None else force_cutoff + self.skin
        rebuild = self._pairs is None or self._list_cutoff is None or abs(self._list_cutoff - list_cutoff) > 1.0e-12
        if not rebuild and not self.freeze_neighbor_list and context.step % self.rebuild_check_every == 0:
            assert self._reference_a is not None and self._reference_b is not None
            displacement = torch.linalg.norm(xa - self._reference_a, dim=1).amax() + torch.linalg.norm(xb - self._reference_b, dim=1).amax()
            rebuild = bool((displacement > self.skin).item())
        if rebuild:
            self._pairs = self._build_pairs(xa, xb, list_cutoff)
            self._reference_a, self._reference_b, self._list_cutoff = xa.detach().clone(), xb.detach().clone(), list_cutoff
            self._rebuild_count += 1
        assert self._pairs is not None
        wa = NodalWrench.zeros(body_a.model.n_node, device=xa.device, dtype=xa.dtype)
        wb = NodalWrench.zeros(body_b.model.n_node, device=xb.device, dtype=xb.dtype)
        energy = xa.new_zeros(()) if diagnostics else None
        n_used = 0
        min_distance = math.inf
        prefactor = charge**2 / (4.0 * math.pi * eps0 * self.relative_permittivity)
        for start in range(0, self._pairs.shape[0], self.force_chunk_size):
            pair = self._pairs[start:start + self.force_chunk_size]
            ia, ib = pair[:, 0], pair[:, 1]
            rvec = xa[ia] - xb[ib]
            raw_r = torch.linalg.norm(rvec, dim=1)
            active = raw_r <= force_cutoff if self.apply_force_cutoff else torch.ones_like(raw_r, dtype=torch.bool)
            r = raw_r.clamp_min(self.min_distance)
            exponential = torch.exp(-kappa * r)
            active_float = active.to(r.dtype)
            magnitude = prefactor * exponential * (kappa / r + 1.0 / r**2) * active_float
            force = magnitude[:, None] * rvec / raw_r.clamp_min(1.0e-12)[:, None]
            wa.force.index_add_(0, ia, force)
            wb.force.index_add_(0, ib, -force)
            if diagnostics:
                assert energy is not None
                energy = energy + (prefactor * exponential / r * active_float).sum()
            if diagnostics:
                n_used += int(active.sum().item())
                min_distance = min(min_distance, float(torch.where(active, raw_r, torch.full_like(raw_r, math.inf)).min().item()))
        diagnostic_data = {}
        if diagnostics:
            diagnostic_data = {
                "n_candidate": int(self._pairs.shape[0]),
                "n_used": n_used,
                "min_distance": min_distance,
                "neighbor_rebuild_count": self._rebuild_count,
                "debye_length": ld,
                "force_cutoff": force_cutoff if self.apply_force_cutoff else math.inf,
                "initial_pair_cutoff": list_cutoff,
                "neighbor_list_frozen": self.freeze_neighbor_list,
                "effective_charge": charge,
                "body_a_net_force_pN": wa.force.sum(dim=0).detach(),
                "body_a_torque_about_center_pN_nm": torch.linalg.cross(body_a.reconstruction.xrel, wa.force, dim=1).sum(dim=0).detach(),
            }
        return InteractionResult(nodal_wrenches={self.body_a: wa, self.body_b: wb}, energy=energy, diagnostics=diagnostic_data)
