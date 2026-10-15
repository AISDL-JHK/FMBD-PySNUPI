"""Structured, human-readable simulation lifecycle logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import torch


def _scalar(value: Any) -> float | None:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass
class SimulationLogWriter:
    """Write a compact lifecycle log to an output directory.

    ``Simulation(output_dir=...)`` registers this writer automatically unless
    the caller already supplied one. When a DCD writer is registered, Simulation
    aligns every log writer with its frame interval (the first DCD writer if
    multiple trajectories are registered). Without a DCD writer, ``every`` is
    used as configured. Diagnostics are calculated on each logged step.
    """

    path: str | Path
    every: int = 500
    _handle: Any = field(default=None, init=False, repr=False)
    _started_at: float | None = field(default=None, init=False, repr=False)

    def _wall_time(self) -> str:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        elapsed = 0.0 if self._started_at is None else time.perf_counter() - self._started_at
        return f"wall_time={timestamp} elapsed_s={elapsed:.3f}"

    def on_start(self, simulation: Any, context: Any) -> None:
        if self.every <= 0:
            raise ValueError("SimulationLogWriter.every must be positive")
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8", buffering=1)
        self._started_at = time.perf_counter()
        self._handle.write("FMBD simulation log\n")
        self._handle.write(f"started {self._wall_time()}\n")
        self._handle.write(f"bodies: {', '.join(simulation.system.bodies)}\n")
        self._handle.write(f"dt_ps: {context.dt}\n")
        self._handle.write("columns: completed_step time_ps wall_time elapsed_s total_energy_pN_nm body_force_norms\n")

    def on_step(self, simulation: Any, result: Any) -> None:
        completed = result.context.step + 1
        if completed % self.every != 0:
            return
        total_energy = sum(value for value in (_scalar(item) for item in result.energies.values()) if value is not None)
        force_norms = []
        for name, values in result.integrator_diagnostics.items():
            ft = _scalar(values.get("Ft_norm"))
            fr = _scalar(values.get("Fr_norm"))
            fq = _scalar(values.get("Fq_norm"))
            force_norms.append(f"{name}:Ft={ft:.3e},Fr={fr:.3e},Fq={fq:.3e}")
        self._handle.write(
            f"step={completed} time_ps={result.context.time + result.context.dt:.6g} "
            f"{self._wall_time()} "
            f"energy_pN_nm={total_energy:.6e} {'; '.join(force_norms)}\n"
        )

    def on_finish(self, simulation: Any) -> None:
        if self._handle is None:
            return
        self._handle.write(f"finished_completed_step={getattr(simulation, 'last_completed_step', None)} {self._wall_time()}\n")
        self._handle.close()
        self._handle = None
