"""System container independent of a particular physical experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from FMBD.core.body import Body
from FMBD.core.reconstruction import reconstruct


@dataclass
class FMBDSystem:
    bodies: dict[str, Body] = field(default_factory=dict)
    interactions: list[Any] = field(default_factory=list)

    def add_body(self, name: str, body: Body) -> None:
        if name in self.bodies:
            raise KeyError(f"body {name!r} is already registered")
        self.bodies[name] = body

    def add_interaction(self, interaction: Any) -> None:
        self.interactions.append(interaction)

    def reconstruct_all(self) -> None:
        for body in self.bodies.values():
            reconstruct(body)
