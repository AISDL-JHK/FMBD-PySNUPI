"""Force containers shared by mechanics, interactions and dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NodalWrench:
    """Forces and moments at a body's nodes, both shaped ``(N, 3)``."""

    force: torch.Tensor
    moment: torch.Tensor

    @classmethod
    def zeros(cls, n_node: int, *, device: torch.device, dtype: torch.dtype) -> "NodalWrench":
        return cls(torch.zeros((n_node, 3), device=device, dtype=dtype), torch.zeros((n_node, 3), device=device, dtype=dtype))

    def add_(self, other: "NodalWrench") -> "NodalWrench":
        self.force.add_(other.force)
        self.moment.add_(other.moment)
        return self


@dataclass
class GeneralizedForce:
    """Body translation, rotation and reduced-modal generalized forces."""

    translation: torch.Tensor
    rotation: torch.Tensor
    modal: torch.Tensor

    @classmethod
    def zeros(cls, n_mode: int, *, device: torch.device, dtype: torch.dtype) -> "GeneralizedForce":
        return cls(torch.zeros(3, device=device, dtype=dtype), torch.zeros(3, device=device, dtype=dtype), torch.zeros(n_mode, device=device, dtype=dtype))

    def add_(self, other: "GeneralizedForce") -> "GeneralizedForce":
        self.translation.add_(other.translation)
        self.rotation.add_(other.rotation)
        self.modal.add_(other.modal)
        return self
