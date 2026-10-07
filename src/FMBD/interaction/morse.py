"""Vectorized pairwise Morse stacking interaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from FMBD.core.forces import NodalWrench
from FMBD.interaction.base import InteractionResult


@dataclass
class MorsePairInteraction:
    """Morse attraction between selected nodes on two named bodies."""

    body_a: str
    body_b: str
    pairs: np.ndarray
    epsilon: float
    a: float
    r0: float
    name: str = "stacking"
    _pairs: torch.Tensor | None = field(default=None, init=False, repr=False)

    def initialize(self, system: Any, context: Any) -> None:
        if self.body_a not in system.bodies or self.body_b not in system.bodies:
            raise KeyError("Morse interaction body names must exist in the system")
        pairs = np.asarray(self.pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (N, 2)")
        a_body, b_body = system.bodies[self.body_a], system.bodies[self.body_b]
        if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= a_body.model.n_node):
            raise IndexError("Morse body_a node index out of range")
        if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= b_body.model.n_node):
            raise IndexError("Morse body_b node index out of range")
        self._pairs = torch.as_tensor(pairs, device=a_body.model.X.device, dtype=torch.long)

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        if self._pairs is None:
            self.initialize(system, context)
        body_a, body_b = system.bodies[self.body_a], system.bodies[self.body_b]
        rec_a, rec_b = body_a.reconstruction, body_b.reconstruction
        if rec_a is None or rec_b is None:
            raise RuntimeError("all interacting bodies must be reconstructed")
        assert self._pairs is not None
        ia, ib = self._pairs[:, 0], self._pairs[:, 1]
        rvec = rec_b.x[ib] - rec_a.x[ia]
        distance = torch.linalg.norm(rvec, dim=1).clamp_min(1.0e-12)
        unit = rvec / distance[:, None]
        z = torch.exp(-self.a * (distance - self.r0))
        energy = (self.epsilon * (1.0 - z) ** 2 - self.epsilon).sum() if diagnostics else None
        radial_force = 2.0 * self.epsilon * self.a * (z - z * z)
        force_a_pair = radial_force[:, None] * unit
        wrench_a = NodalWrench.zeros(body_a.model.n_node, device=rec_a.x.device, dtype=rec_a.x.dtype)
        wrench_b = NodalWrench.zeros(body_b.model.n_node, device=rec_b.x.device, dtype=rec_b.x.dtype)
        wrench_a.force.index_add_(0, ia, force_a_pair)
        wrench_b.force.index_add_(0, ib, -force_a_pair)
        result = InteractionResult(nodal_wrenches={self.body_a: wrench_a, self.body_b: wrench_b}, energy=energy)
        if diagnostics:
            result.diagnostics = {"distances": distance.detach(), "radial_forces": radial_force.detach()}
        return result
