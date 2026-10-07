"""Force containers shared by mechanics, interactions and dynamics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodalWrench:
    force: object
    moment: object

    @classmethod
    def zeros(cls, n_node: int, *, xp, dtype):
        return cls(xp.zeros((n_node, 3), dtype=dtype), xp.zeros((n_node, 3), dtype=dtype))

    def add_(self, other: "NodalWrench"):
        self.force += other.force
        self.moment += other.moment
        return self


@dataclass
class GeneralizedForce:
    translation: object
    rotation: object
    modal: object

    @classmethod
    def zeros(cls, n_mode: int, *, xp, dtype):
        return cls(xp.zeros(3, dtype=dtype), xp.zeros(3, dtype=dtype), xp.zeros(n_mode, dtype=dtype))

    def add_(self, other: "GeneralizedForce"):
        self.translation += other.translation
        self.rotation += other.rotation
        self.modal += other.modal
        return self


__all__ = ["GeneralizedForce", "NodalWrench"]
