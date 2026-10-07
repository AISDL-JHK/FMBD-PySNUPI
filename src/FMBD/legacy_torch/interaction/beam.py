"""Orientation-aware Euler--Bernoulli beams between nodes of two bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from FMBD.legacy_torch.core.forces import GeneralizedForce, NodalWrench
from FMBD.legacy_torch.interaction.base import InteractionResult
from FMBD.legacy_torch.math.so3 import log_so3


def _skew_batch(v: torch.Tensor) -> torch.Tensor:
    """Cross-product matrices for a ``(B, 3)`` batch of vectors."""

    K = v.new_zeros((v.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    return K


def _so3_jacobian_inverse_batch(phi: torch.Tensor, *, right: bool) -> torch.Tensor:
    theta2 = torch.sum(phi * phi, dim=1)
    theta = torch.sqrt(theta2.clamp_min(1.0e-30))
    K = _skew_batch(phi)
    regular = 1.0 / theta2.clamp_min(1.0e-30) - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta).clamp_min(1.0e-30))
    series = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    coefficient = torch.where(theta2 < 1.0e-8, series, regular)[:, None, None]
    identity = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(phi.shape[0], 3, 3)
    return identity + (0.5 if right else -0.5) * K + coefficient * torch.bmm(K, K)


def _log_so3_batch(R: torch.Tensor) -> torch.Tensor:
    cos_theta = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack((R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]), dim=1)
    regular = 0.5 * theta[:, None] / torch.sin(theta).clamp_min(1.0e-12)[:, None] * vee
    return torch.where((theta.abs() < 1.0e-7)[:, None], 0.5 * vee, regular)


def _connector_frame(direction: torch.Tensor, node_frame: torch.Tensor) -> torch.Tensor:
    ez = direction / torch.linalg.norm(direction).clamp_min(1.0e-12)
    ex = node_frame[:, 0] - torch.dot(node_frame[:, 0], ez) * ez
    if torch.linalg.norm(ex).item() < 1.0e-8:
        ex = node_frame[:, 1] - torch.dot(node_frame[:, 1], ez) * ez
    ex = ex / torch.linalg.norm(ex).clamp_min(1.0e-12)
    ey = torch.linalg.cross(ez, ex, dim=0)
    ey = ey / torch.linalg.norm(ey).clamp_min(1.0e-12)
    return torch.stack((torch.linalg.cross(ey, ez, dim=0), ey, ez), dim=1)


@dataclass(frozen=True)
class _BatchedBeamReference:
    node_a: torch.Tensor
    node_b: torch.Tensor
    length: torch.Tensor
    frame_a_local: torch.Tensor
    frame_b_local: torch.Tensor
    displacement_a_local: torch.Tensor
    relative_frame: torch.Tensor


@dataclass
class EulerBernoulliBeamInteraction:
    """Six zero-strain, orientation-aware inter-body Euler--Bernoulli beams.

    ``pairs`` are ``[body_a node, body_b node]`` indices.  Reference lengths,
    connector frames, and relative node-frame orientations are captured at
    initialization, making the supplied initial placement exactly zero strain.
    """

    body_a: str
    body_b: str
    pairs: Sequence[Sequence[int]]
    EA: float
    EI1: float
    EI2: float
    GJ: float
    name: str = "euler_bernoulli_beams"
    _references: _BatchedBeamReference | None = None

    def initialize(self, system: Any, context: Any) -> None:
        if self.body_a not in system.bodies or self.body_b not in system.bodies:
            raise KeyError("beam body names must exist in the system")
        pair_array = np.asarray(self.pairs, dtype=np.int64)
        if pair_array.ndim != 2 or pair_array.shape[1] != 2 or len(pair_array) == 0:
            raise ValueError("beam pairs must have shape (n_beam, 2)")
        a, b = system.bodies[self.body_a], system.bodies[self.body_b]
        if pair_array[:, 0].min() < 0 or pair_array[:, 0].max() >= a.model.n_node or pair_array[:, 1].min() < 0 or pair_array[:, 1].max() >= b.model.n_node:
            raise IndexError("beam node index is out of range")

    def _build_references(self, body_a: Any, body_b: Any) -> _BatchedBeamReference:
        assert body_a.reconstruction is not None and body_b.reconstruction is not None
        node_a, node_b, length, frame_a, frame_b, displacement, relative = [], [], [], [], [], [], []
        for a_index, b_index in np.asarray(self.pairs, dtype=np.int64):
            a, b = int(a_index), int(b_index)
            direction = body_b.reconstruction.x[b] - body_a.reconstruction.x[a]
            frame = _connector_frame(direction, body_a.reconstruction.Qglobal[a])
            Ca = body_a.reconstruction.Qglobal[a].T @ frame
            Cb = body_b.reconstruction.Qglobal[b].T @ frame
            Ha, Hb = body_a.reconstruction.Qglobal[a] @ Ca, body_b.reconstruction.Qglobal[b] @ Cb
            node_a.append(a)
            node_b.append(b)
            length.append(torch.linalg.norm(direction).detach())
            frame_a.append(Ca.detach())
            frame_b.append(Cb.detach())
            displacement.append((Ha.T @ direction).detach())
            relative.append((Ha.T @ Hb).detach())
        return _BatchedBeamReference(
            torch.tensor(node_a, device=body_a.model.X.device, dtype=torch.long),
            torch.tensor(node_b, device=body_b.model.X.device, dtype=torch.long),
            torch.stack(length), torch.stack(frame_a), torch.stack(frame_b), torch.stack(displacement), torch.stack(relative),
        )

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        body_a, body_b = system.bodies[self.body_a], system.bodies[self.body_b]
        assert body_a.reconstruction is not None and body_b.reconstruction is not None
        if self._references is None:
            self._references = self._build_references(body_a, body_b)
        wa = NodalWrench.zeros(body_a.model.n_node, device=body_a.model.X.device, dtype=body_a.model.X.dtype)
        wb = NodalWrench.zeros(body_b.model.n_node, device=body_b.model.X.device, dtype=body_b.model.X.dtype)
        energy = body_a.model.X.new_zeros(()) if diagnostics else None
        torque_on_a = body_a.model.X.new_zeros(3) if diagnostics else None
        ref = self._references
        xa, xb = body_a.reconstruction.x[ref.node_a], body_b.reconstruction.x[ref.node_b]
        Qa, Qb = body_a.reconstruction.Qglobal[ref.node_a], body_b.reconstruction.Qglobal[ref.node_b]
        Ha, Hb = torch.bmm(Qa, ref.frame_a_local), torch.bmm(Qb, ref.frame_b_local)
        direction = xb - xa
        displacement = torch.bmm(Ha.transpose(1, 2), direction[:, :, None]).squeeze(2) - ref.displacement_a_local
        theta = _log_so3_batch(torch.bmm(ref.relative_frame.transpose(1, 2), torch.bmm(Ha.transpose(1, 2), Hb)))
        L = ref.length.clamp_min(1.0e-12)
        force_local = torch.stack(((12.0 * self.EI2 / L**3) * displacement[:, 0] - (6.0 * self.EI2 / L**2) * theta[:, 1], (12.0 * self.EI1 / L**3) * displacement[:, 1] + (6.0 * self.EI1 / L**2) * theta[:, 0], (self.EA / L) * displacement[:, 2]), dim=1)
        force_global = torch.bmm(Ha, force_local[:, :, None]).squeeze(2)
        moment_theta = torch.stack(((6.0 * self.EI1 / L**2) * displacement[:, 1] + (4.0 * self.EI1 / L) * theta[:, 0], -(6.0 * self.EI2 / L**2) * displacement[:, 0] + (4.0 * self.EI2 / L) * theta[:, 1], (self.GJ / L) * theta[:, 2]), dim=1)
        left = _so3_jacobian_inverse_batch(theta, right=False)
        right = _so3_jacobian_inverse_batch(theta, right=True)
        moment_a = torch.linalg.cross(direction, force_global, dim=1) + torch.bmm(Ha, torch.bmm(ref.relative_frame, torch.bmm(left.transpose(1, 2), moment_theta[:, :, None]))).squeeze(2)
        moment_b = -torch.bmm(Hb, torch.bmm(right.transpose(1, 2), moment_theta[:, :, None])).squeeze(2)
        wa.force.index_add_(0, ref.node_a, force_global)
        wb.force.index_add_(0, ref.node_b, -force_global)
        wa.moment.index_add_(0, ref.node_a, moment_a)
        wb.moment.index_add_(0, ref.node_b, moment_b)
        records: list[dict[str, torch.Tensor]] = []
        if diagnostics:
            assert energy is not None and torque_on_a is not None
            torque_on_a = (torch.linalg.cross(xa - body_a.state.c, force_global, dim=1) + moment_a).sum(dim=0)
            axial = 0.5 * (self.EA / L) * displacement[:, 2] ** 2
            bend1 = 0.5 * ((12.0 * self.EI1 / L**3) * displacement[:, 1] ** 2 + (12.0 * self.EI1 / L**2) * displacement[:, 1] * theta[:, 0] + (4.0 * self.EI1 / L) * theta[:, 0] ** 2)
            bend2 = 0.5 * ((12.0 * self.EI2 / L**3) * displacement[:, 0] ** 2 - (12.0 * self.EI2 / L**2) * displacement[:, 0] * theta[:, 1] + (4.0 * self.EI2 / L) * theta[:, 1] ** 2)
            energy = (axial + bend1 + bend2 + 0.5 * (self.GJ / L) * theta[:, 2] ** 2).sum()
            records = [{"length_nm": torch.linalg.norm(direction[i]).detach(), "displacement_local_nm": displacement[i].detach(), "rotation_local_rad": theta[i].detach(), "force_global_pN": force_global[i].detach(), "moment_a_pN_nm": moment_a[i].detach()} for i in range(len(ref.node_a))]
        detail = {"beams": records, "torque_on_body_a_pN_nm": torque_on_a.detach()} if diagnostics else {}
        return InteractionResult(nodal_wrenches={self.body_a: wa, self.body_b: wb}, energy=energy, diagnostics=detail)


@dataclass
class CollectiveAxialTorsionInteraction:
    """Fast collective torsional coupling of two bodies about one connector axis.

    Individual beams resolve local extension and bending.  This O(1)
    interaction captures the six-crossover bundle's collective yaw stiffness
    directly in body coordinates, preventing a reduced modal basis from
    absorbing the rotational transfer path.
    """

    body_a: str
    body_b: str
    axis_global: torch.Tensor
    stiffness_pN_nm_per_rad: float
    name: str = "collective_connector_torsion"
    _rotation_a0: torch.Tensor | None = None
    _rotation_b0: torch.Tensor | None = None
    _axis: torch.Tensor | None = None

    def initialize(self, system: Any, context: Any) -> None:
        if self.body_a not in system.bodies or self.body_b not in system.bodies:
            raise KeyError("torsional-joint body names must exist in the system")
        if self.stiffness_pN_nm_per_rad < 0.0:
            raise ValueError("torsional-joint stiffness must be non-negative")
        a, b = system.bodies[self.body_a], system.bodies[self.body_b]
        self._rotation_a0 = a.state.R.detach().clone()
        self._rotation_b0 = b.state.R.detach().clone()
        axis = torch.as_tensor(self.axis_global, device=a.state.R.device, dtype=a.state.R.dtype)
        self._axis = axis / torch.linalg.norm(axis).clamp_min(1.0e-12)

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        if self._rotation_a0 is None or self._rotation_b0 is None or self._axis is None:
            raise RuntimeError("collective torsional joint was not initialized")
        a, b = system.bodies[self.body_a], system.bodies[self.body_b]
        relative_change = (b.state.R @ self._rotation_b0.T) @ (a.state.R @ self._rotation_a0.T).T
        twist = torch.dot(log_so3(relative_change), self._axis)
        torque = self.stiffness_pN_nm_per_rad * twist * self._axis
        force_a = GeneralizedForce.zeros(a.model.n_mode, device=a.state.q.device, dtype=a.state.q.dtype)
        force_b = GeneralizedForce.zeros(b.model.n_mode, device=b.state.q.device, dtype=b.state.q.dtype)
        force_a.rotation += torque
        force_b.rotation -= torque
        energy = 0.5 * self.stiffness_pN_nm_per_rad * twist**2 if diagnostics else None
        detail = {"twist_rad": twist.detach(), "torque_on_a_pN_nm": torque.detach()} if diagnostics else {}
        return InteractionResult(generalized_forces={self.body_a: force_a, self.body_b: force_b}, energy=energy, diagnostics=detail)
