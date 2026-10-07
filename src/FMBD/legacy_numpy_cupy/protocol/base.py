"""Protocol context shared with interactions and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimulationContext:
    step: int
    time: float
    dt: float
    temperature: float | None = None
    kBT: float | None = None
    environment: dict[str, Any] = field(default_factory=dict)
