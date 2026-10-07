"""System-independent FMBD simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from FMBD.legacy_numpy_cupy.core.elasticity import linear_reduced_force_and_energy
from FMBD.legacy_numpy_cupy.core.forces import GeneralizedForce, NodalWrench
from FMBD.legacy_numpy_cupy.core.projection import project_nodal_wrench


@dataclass
class StepResult:
    context: Any
    forces: dict[str, GeneralizedForce]
    energies: dict[str, Any]
    interaction_diagnostics: dict[str, dict[str, Any]]
    integrator_diagnostics: dict[str, dict[str, Any]]


@dataclass
class Simulation:
    system: Any
    protocol: Any
    integrator: Any
    observers: list[Any] | tuple[Any, ...] = ()
    output_dir: str | Path | None = None
    _initialized: bool = False
    last_completed_step: int | None = None

    def __post_init__(self) -> None:
        """Attach the framework-default output log when an output directory is given."""
        self.observers = list(self.observers)
        if self.output_dir is None:
            return
        from FMBD.legacy_numpy_cupy.observer.log import SimulationLogWriter
        if not any(isinstance(observer, SimulationLogWriter) for observer in self.observers):
            self.observers.append(SimulationLogWriter(Path(self.output_dir) / "simulation.log"))

    def initialize(self) -> None:
        self.system.reconstruct_all()
        context0 = self.protocol.context(0)
        for interaction in self.system.interactions:
            interaction.initialize(self.system, context0)
        for observer in self.observers:
            observer.on_start(self, context0)
        self._initialized = True

    def evaluate_forces(self, context, diagnostics: bool = False) -> tuple[dict[str, GeneralizedForce], dict[str, Any], dict[str, dict[str, Any]]]:
        nodal = {name: NodalWrench.zeros(body.model.n_node, xp=body.model.xp, dtype=body.model.dtype) for name, body in self.system.bodies.items()}
        forces = {name: GeneralizedForce.zeros(body.model.n_mode, xp=body.model.xp, dtype=body.model.dtype) for name, body in self.system.bodies.items()}
        energies: dict[str, Any] = {}
        for name, body in self.system.bodies.items():
            restoring_force, strain_energy = linear_reduced_force_and_energy(body, diagnostics=diagnostics)
            forces[name].modal += restoring_force
            if strain_energy is not None:
                energies[f"body.{name}.strain"] = strain_energy
        interaction_diagnostics: dict[str, dict[str, Any]] = {}
        for interaction in self.system.interactions:
            result = interaction.compute(self.system, context, diagnostics=diagnostics)
            for name, wrench in result.nodal_wrenches.items():
                nodal[name].add_(wrench)
            for name, force in result.generalized_forces.items():
                forces[name].add_(force)
            if result.energy is not None:
                if interaction.name in energies:
                    energies[interaction.name] = energies[interaction.name] + result.energy
                else:
                    energies[interaction.name] = result.energy
            if diagnostics:
                interaction_diagnostics[interaction.name] = result.diagnostics
        for name, body in self.system.bodies.items():
            forces[name].add_(project_nodal_wrench(body, nodal[name]))
        return forces, energies, interaction_diagnostics

    def step(self, step: int, diagnostics: bool = False) -> StepResult:
        if not self._initialized:
            self.initialize()
        context = self.protocol.context(step)
        forces, energies, interaction_diagnostics = self.evaluate_forces(context, diagnostics)
        integrator_diagnostics = self.integrator.step(
            self.system, forces, context,
            force_evaluator=lambda midpoint_context: self.evaluate_forces(midpoint_context)[0],
        )
        self.system.reconstruct_all()
        self.last_completed_step = step + 1
        result = StepResult(context, forces, energies, interaction_diagnostics, integrator_diagnostics)
        for observer in self.observers:
            observer.on_step(self, result)
        return result

    def finish(self) -> None:
        for observer in self.observers:
            observer.on_finish(self)

    def run(
        self,
        n_steps: int | None = None,
        *,
        diagnostics_every: int | None = None,
        callback: Callable[[StepResult], None] | None = None,
    ) -> None:
        """Run a finite protocol and always finalize registered observers.

        ``callback`` receives every completed step and is the appropriate place
        for application-specific logging or checkpoints.
        """
        if n_steps is None:
            try:
                n_steps = self.protocol.n_steps
            except AttributeError as exc:
                raise ValueError("n_steps is required for a protocol without n_steps") from exc
        try:
            for step in range(n_steps):
                diagnostics = diagnostics_every is not None and (step + 1) % diagnostics_every == 0
                result = self.step(step, diagnostics=diagnostics)
                if callback is not None:
                    callback(result)
        finally:
            self.finish()
