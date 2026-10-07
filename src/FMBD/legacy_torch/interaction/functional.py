"""Adapter for a small, function-defined interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from FMBD.legacy_torch.interaction.base import InteractionResult


@dataclass
class FunctionalInteraction:
    function: Callable[[Any, Any, bool], InteractionResult]
    name: str = "functional"

    def initialize(self, system: Any, context: Any) -> None:
        return None

    def compute(self, system: Any, context: Any, diagnostics: bool = False) -> InteractionResult:
        return self.function(system, context, diagnostics)
