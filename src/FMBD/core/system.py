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
    batch_bodies: bool = True
    batch_linkers: bool = True

    def add_body(self, name: str, body: Body) -> None:
        if name in self.bodies:
            raise KeyError(f"body {name!r} is already registered")
        self.bodies[name] = body

    def add_interaction(self, interaction: Any) -> None:
        self.interactions.append(interaction)

    def reconstruct_all(self) -> None:
        if not self.batch_bodies:
            for body in self.bodies.values():
                reconstruct(body)
            return
        from FMBD.core.batching import grouped_bodies
        from FMBD.core.reconstruction import reconstruct_batch
        for group in grouped_bodies(self):
            if len(group.bodies)==1:
                reconstruct(group.bodies[0])
            else:
                reconstruct_batch(group)
