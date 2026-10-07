"""Structured, human-readable simulation lifecycle logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    the caller already supplied one.  The default cadence is deliberately
    sparse so long Brownian-dynamics trajectories do not produce huge logs.
    """

    path: str | Path
    every: int = 100
    _handle: Any = field(default=None, init=False, repr=False)

    def on_start(self, simulation: Any, context: Any) -> None:
        if self.every <= 0:
            raise ValueError("SimulationLogWriter.every must be positive")
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8", buffering=1)
        self._handle.write("FMBD simulation log\n")
        self._handle.write(f"bodies: {', '.join(simulation.system.bodies)}\n")
        self._handle.write(f"dt_ps: {context.dt}\n")
        self._handle.write("columns: completed_step time_ps total_energy_pN_nm body_force_norms\n")

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
            f"energy_pN_nm={total_energy:.6e} {'; '.join(force_norms)}\n"
        )

    def on_finish(self, simulation: Any) -> None:
        if self._handle is None:
            return
        self._handle.write(f"finished_completed_step={getattr(simulation, 'last_completed_step', None)}\n")
        self._handle.close()
        self._handle = None
