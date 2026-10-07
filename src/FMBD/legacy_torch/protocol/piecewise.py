"""A finite piecewise-constant protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from FMBD.legacy_torch.protocol.base import SimulationContext


@dataclass(frozen=True)
class PiecewiseProtocol:
    """Expose one piecewise constant value through ``context.environment``."""

    key: str
    schedule: Sequence[tuple[float, int]]
    dt: float
    temperature: float | None = None
    kBT: float | None = None

    def context(self, step: int) -> SimulationContext:
        if step < 0:
            raise ValueError("step must be non-negative")
        cumulative = 0
        for index, (value, n_steps) in enumerate(self.schedule, start=1):
            if n_steps <= 0:
                raise ValueError("piecewise protocol durations must be positive")
            cumulative += n_steps
            if step < cumulative:
                return SimulationContext(
                    step=step, time=step * self.dt, dt=self.dt,
                    temperature=self.temperature, kBT=self.kBT,
                    environment={self.key: value, "stage_index": index, "stage_step": step - (cumulative - n_steps) + 1, "stage_n_steps": n_steps},
                )
        raise IndexError(f"step {step} is beyond the protocol duration ({cumulative})")

    @property
    def n_steps(self) -> int:
        return sum(n_steps for _, n_steps in self.schedule)
