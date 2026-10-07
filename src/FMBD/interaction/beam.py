"""Orientation-aware Euler-Bernoulli connector beams on NumPy/CuPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from FMBD.core.forces import GeneralizedForce, NodalWrench
from FMBD.interaction.base import InteractionResult
from FMBD.math.so3 import log_so3


def _skew_batch(v, xp):
    k = xp.zeros((len(v), 3, 3), dtype=v.dtype)
    k[:, 0, 1], k[:, 0, 2] = -v[:, 2], v[:, 1]
    k[:, 1, 0], k[:, 1, 2] = v[:, 2], -v[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -v[:, 1], v[:, 0]
    return k


def _so3_jacobian_inverse_batch(phi, xp, *, right):
    theta2 = xp.sum(phi * phi, axis=1)
    theta = xp.sqrt(xp.maximum(theta2, 1.0e-30))
    k = _skew_batch(phi, xp)
    regular = 1.0 / xp.maximum(theta2, 1.0e-30) - (1.0 + xp.cos(theta)) / xp.maximum(2.0 * theta * xp.sin(theta), 1.0e-30)
    series = 1.0 / 12.0 + theta2 / 720.0 + theta2**2 / 30240.0
    coeff = xp.where(theta2 < 1.0e-8, series, regular)[:, None, None]
    identity = xp.broadcast_to(xp.eye(3, dtype=phi.dtype), (len(phi), 3, 3))
    return identity + (0.5 if right else -0.5) * k + coeff * (k @ k)


def _log_so3_batch(R, xp):
    cosine = xp.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    theta = xp.arccos(cosine)
    vee = xp.stack((R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]), axis=1)
    regular = 0.5 * theta[:, None] / xp.maximum(xp.abs(xp.sin(theta)), 1.0e-12)[:, None] * vee
    return xp.where((xp.abs(theta) < 1.0e-7)[:, None], 0.5 * vee, regular)


def _connector_frame(direction, node_frame, xp):
    ez = direction / xp.maximum(xp.linalg.norm(direction), 1.0e-12)
    ex = node_frame[:, 0] - xp.dot(node_frame[:, 0], ez) * ez
    if float(xp.linalg.norm(ex).item()) < 1.0e-8:
        ex = node_frame[:, 1] - xp.dot(node_frame[:, 1], ez) * ez
    ex /= xp.maximum(xp.linalg.norm(ex), 1.0e-12)
    ey = xp.cross(ez, ex)
    ey /= xp.maximum(xp.linalg.norm(ey), 1.0e-12)
    return xp.stack((xp.cross(ey, ez), ey, ez), axis=1)


@dataclass(frozen=True)
class _BatchedBeamReference:
    node_a: Any
    node_b: Any
    length: Any
    frame_a_local: Any
    frame_b_local: Any
    displacement_a_local: Any
    relative_frame: Any


@dataclass
class EulerBernoulliBeamInteraction:
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
        pairs = np.asarray(self.pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
            raise ValueError("beam pairs must have shape (n_beam, 2)")
        a, b = system.bodies[self.body_a], system.bodies[self.body_b]
        if a.model.backend != b.model.backend:
            raise ValueError("interacting bodies must use the same array backend")
        if pairs[:, 0].min() < 0 or pairs[:, 0].max() >= a.model.n_node or pairs[:, 1].min() < 0 or pairs[:, 1].max() >= b.model.n_node:
            raise IndexError("beam node index is out of range")

    def _build_references(self, body_a, body_b):
        xp = body_a.model.xp
        rec_a, rec_b = body_a.reconstruction, body_b.reconstruction
        pairs = np.asarray(self.pairs, dtype=np.int64)
        ia, ib, length, frame_a, frame_b, displacement, relative = [], [], [], [], [], [], []
        for ai, bi in pairs:
            a, b = int(ai), int(bi)
            direction = rec_b.x[b] - rec_a.x[a]
            frame = _connector_frame(direction, rec_a.Qglobal[a], xp)
            ca, cb = rec_a.Qglobal[a].T @ frame, rec_b.Qglobal[b].T @ frame
            ha, hb = rec_a.Qglobal[a] @ ca, rec_b.Qglobal[b] @ cb
            ia.append(a); ib.append(b); length.append(xp.linalg.norm(direction))
            frame_a.append(ca); frame_b.append(cb); displacement.append(ha.T @ direction); relative.append(ha.T @ hb)
        return _BatchedBeamReference(
            xp.asarray(ia, dtype=xp.int64), xp.asarray(ib, dtype=xp.int64), xp.stack(length),
            xp.stack(frame_a), xp.stack(frame_b), xp.stack(displacement), xp.stack(relative),
        )

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        a, b = system.bodies[self.body_a], system.bodies[self.body_b]
        if a.reconstruction is None or b.reconstruction is None:
            raise RuntimeError("all interacting bodies must be reconstructed")
        xp = a.model.xp
        if self._references is None:
            self._references = self._build_references(a, b)
        wa = NodalWrench.zeros(a.model.n_node, xp=xp, dtype=a.model.dtype)
        wb = NodalWrench.zeros(b.model.n_node, xp=xp, dtype=b.model.dtype)
        ref = self._references
        xa, xb = a.reconstruction.x[ref.node_a], b.reconstruction.x[ref.node_b]
        qa, qb = a.reconstruction.Qglobal[ref.node_a], b.reconstruction.Qglobal[ref.node_b]
        ha, hb = qa @ ref.frame_a_local, qb @ ref.frame_b_local
        direction = xb - xa
        displacement = (ha.transpose(0, 2, 1) @ direction[:, :, None]).squeeze(2) - ref.displacement_a_local
        theta = _log_so3_batch(ref.relative_frame.transpose(0, 2, 1) @ (ha.transpose(0, 2, 1) @ hb), xp)
        length = xp.maximum(ref.length, 1.0e-12)
        force_local = xp.stack(((12*self.EI2/length**3)*displacement[:, 0]-(6*self.EI2/length**2)*theta[:, 1], (12*self.EI1/length**3)*displacement[:, 1]+(6*self.EI1/length**2)*theta[:, 0], (self.EA/length)*displacement[:, 2]), axis=1)
        force_global = (ha @ force_local[:, :, None]).squeeze(2)
        moment_theta = xp.stack(((6*self.EI1/length**2)*displacement[:, 1]+(4*self.EI1/length)*theta[:, 0], -(6*self.EI2/length**2)*displacement[:, 0]+(4*self.EI2/length)*theta[:, 1], (self.GJ/length)*theta[:, 2]), axis=1)
        left, right = _so3_jacobian_inverse_batch(theta, xp, right=False), _so3_jacobian_inverse_batch(theta, xp, right=True)
        moment_a = xp.cross(direction, force_global, axis=1) + (ha @ ref.relative_frame @ (left.transpose(0, 2, 1) @ moment_theta[:, :, None])).squeeze(2)
        moment_b = -(hb @ (right.transpose(0, 2, 1) @ moment_theta[:, :, None])).squeeze(2)
        xp.add.at(wa.force, ref.node_a, force_global); xp.add.at(wb.force, ref.node_b, -force_global)
        xp.add.at(wa.moment, ref.node_a, moment_a); xp.add.at(wb.moment, ref.node_b, moment_b)
        energy, records, torque = None, [], None
        if diagnostics:
            torque = xp.sum(xp.cross(xa - a.state.c, force_global, axis=1) + moment_a, axis=0)
            axial = 0.5 * (self.EA/length)*displacement[:, 2]**2
            bend1 = 0.5*((12*self.EI1/length**3)*displacement[:, 1]**2+(12*self.EI1/length**2)*displacement[:, 1]*theta[:, 0]+(4*self.EI1/length)*theta[:, 0]**2)
            bend2 = 0.5*((12*self.EI2/length**3)*displacement[:, 0]**2-(12*self.EI2/length**2)*displacement[:, 0]*theta[:, 1]+(4*self.EI2/length)*theta[:, 1]**2)
            energy = xp.sum(axial+bend1+bend2+0.5*(self.GJ/length)*theta[:, 2]**2)
            records = [{"length_nm": xp.linalg.norm(direction[i]), "displacement_local_nm": displacement[i].copy(), "rotation_local_rad": theta[i].copy(), "force_global_pN": force_global[i].copy(), "moment_a_pN_nm": moment_a[i].copy()} for i in range(len(ref.node_a))]
        return InteractionResult(nodal_wrenches={self.body_a: wa, self.body_b: wb}, energy=energy, diagnostics={"beams": records, "torque_on_body_a_pN_nm": torque} if diagnostics else {})


@dataclass
class CollectiveAxialTorsionInteraction:
    body_a: str
    body_b: str
    axis_global: Sequence[float]
    stiffness_pN_nm_per_rad: float
    name: str = "collective_connector_torsion"
    _rotation_a0: Any = None
    _rotation_b0: Any = None
    _axis: Any = None

    def initialize(self, system: Any, context: Any) -> None:
        a, b = system.bodies[self.body_a], system.bodies[self.body_b]
        if self.stiffness_pN_nm_per_rad < 0.0:
            raise ValueError("torsional-joint stiffness must be non-negative")
        xp = a.model.xp
        self._rotation_a0, self._rotation_b0 = a.state.R.copy(), b.state.R.copy()
        axis = xp.asarray(self.axis_global, dtype=a.model.dtype)
        self._axis = axis / xp.maximum(xp.linalg.norm(axis), 1.0e-12)

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        if self._rotation_a0 is None or self._rotation_b0 is None or self._axis is None:
            raise RuntimeError("collective torsional joint was not initialized")
        a, b = system.bodies[self.body_a], system.bodies[self.body_b]
        xp = a.model.xp
        relative = (b.state.R @ self._rotation_b0.T) @ (a.state.R @ self._rotation_a0.T).T
        twist = xp.dot(log_so3(relative), self._axis)
        torque = self.stiffness_pN_nm_per_rad * twist * self._axis
        force_a = GeneralizedForce.zeros(a.model.n_mode, xp=xp, dtype=a.model.dtype)
        force_b = GeneralizedForce.zeros(b.model.n_mode, xp=xp, dtype=b.model.dtype)
        force_a.rotation += torque; force_b.rotation -= torque
        energy = 0.5*self.stiffness_pN_nm_per_rad*twist**2 if diagnostics else None
        return InteractionResult(generalized_forces={self.body_a: force_a, self.body_b: force_b}, energy=energy, diagnostics={"twist_rad": twist, "torque_on_a_pN_nm": torque} if diagnostics else {})
