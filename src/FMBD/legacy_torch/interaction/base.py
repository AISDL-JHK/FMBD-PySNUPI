"""Interaction protocol and result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from FMBD.legacy_torch.core.forces import GeneralizedForce, NodalWrench


@dataclass
class InteractionResult:
    """One interaction's force contributions, energy and optional diagnostics."""

    nodal_wrenches: dict[str, NodalWrench] = field(default_factory=dict)
    generalized_forces: dict[str, GeneralizedForce] = field(default_factory=dict)
    energy: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Interaction(Protocol):
    """Protocol implemented by stateless and stateful physical interactions."""

    name: str

    def initialize(self, system: Any, context: Any) -> None:
        """Initialize state that depends on the initial reconstructed geometry."""

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        """Return contributions for the system's current geometry."""
