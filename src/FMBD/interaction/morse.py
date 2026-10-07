"""Vectorized pairwise Morse interaction using NumPy or CuPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from FMBD.core.forces import NodalWrench
from FMBD.interaction.base import InteractionResult


@dataclass
class MorsePairInteraction:
    body_a: str
    body_b: str
    pairs: np.ndarray
    epsilon: float
    a: float
    r0: float
    name: str = "stacking"
    _pairs: Any = None

    def initialize(self, system: Any, context: Any) -> None:
        if self.body_a not in system.bodies or self.body_b not in system.bodies:
            raise KeyError("Morse interaction body names must exist in the system")
        pairs = np.asarray(self.pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (N, 2)")
        a_body, b_body = system.bodies[self.body_a], system.bodies[self.body_b]
        if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= a_body.model.n_node) or np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= b_body.model.n_node):
            raise IndexError("Morse interaction node index out of range")
        xp = a_body.model.xp
        if xp is not b_body.model.xp:
            raise ValueError("interacting bodies must use the same array backend")
        self._pairs = xp.asarray(pairs, dtype=xp.int64)

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        if self._pairs is None:
            self.initialize(system, context)
        body_a, body_b = system.bodies[self.body_a], system.bodies[self.body_b]
        rec_a, rec_b = body_a.reconstruction, body_b.reconstruction
        if rec_a is None or rec_b is None:
            raise RuntimeError("all interacting bodies must be reconstructed")
        xp = body_a.model.xp
        ia, ib = self._pairs[:, 0], self._pairs[:, 1]
        rvec = rec_b.x[ib] - rec_a.x[ia]
        distance = xp.maximum(xp.linalg.norm(rvec, axis=1), 1.0e-12)
        unit = rvec / distance[:, None]
        z = xp.exp(-self.a * (distance - self.r0))
        energy = xp.sum(self.epsilon * (1.0 - z) ** 2 - self.epsilon) if diagnostics else None
        radial_force = 2.0 * self.epsilon * self.a * (z - z * z)
        force = radial_force[:, None] * unit
        wa = NodalWrench.zeros(body_a.model.n_node, xp=xp, dtype=rec_a.x.dtype)
        wb = NodalWrench.zeros(body_b.model.n_node, xp=xp, dtype=rec_b.x.dtype)
        xp.add.at(wa.force, ia, force)
        xp.add.at(wb.force, ib, -force)
        result = InteractionResult(nodal_wrenches={self.body_a: wa, self.body_b: wb}, energy=energy)
        if diagnostics:
            result.diagnostics = {"distances": distance, "radial_forces": radial_force}
        return result
