"""System-independent FMBD simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from FMBD.core.elasticity import linear_reduced_force_and_energy
from FMBD.core.forces import GeneralizedForce, NodalWrench
from FMBD.core.projection import project_nodal_wrench
from FMBD.core.projection import project_nodal_wrenches_batch
from FMBD.core.batching import grouped_bodies
import torch


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
        from FMBD.observer.log import SimulationLogWriter
        if not any(isinstance(observer, SimulationLogWriter) for observer in self.observers):
            self.observers.append(SimulationLogWriter(Path(self.output_dir) / "simulation.log"))

    def initialize(self) -> None:
        from FMBD.observer.trajectory import DCDTrajectoryWriter
        from FMBD.observer.log import SimulationLogWriter
        trajectory = next((observer for observer in self.observers
                           if isinstance(observer, DCDTrajectoryWriter)), None)
        if trajectory is not None:
            for observer in self.observers:
                if isinstance(observer, SimulationLogWriter):
                    observer.every = trajectory.every
        self.system.reconstruct_all()
        context0 = self.protocol.context(self.last_completed_step or 0)
        for interaction in self.system.interactions:
            interaction.initialize(self.system, context0)
        for observer in self.observers:
            observer.on_start(self, context0)
        self._initialized = True

    def evaluate_forces(self, context, diagnostics: bool = False) -> tuple[dict[str, GeneralizedForce], dict[str, Any], dict[str, dict[str, Any]]]:
        energies: dict[str, Any] = {}
        batching=getattr(self.system,'batch_bodies',False) and len(self.system.bodies)>1
        if batching:
            nodal,forces={},{}
            groups=grouped_bodies(self.system)
            for group in groups:
                model=group.bodies[0].model
                q=group.state('q')
                modal=-group.apply('K',q)
                nf=model.X.new_zeros((len(group.bodies),model.n_node,3))
                nm=torch.zeros_like(nf)
                ft=model.X.new_zeros((len(group.bodies),3))
                fr=torch.zeros_like(ft)
                strain=-0.5*(q*modal).sum(dim=1) if diagnostics else None
                for i,name in enumerate(group.names):
                    nodal[name]=NodalWrench(nf[i],nm[i])
                    forces[name]=GeneralizedForce(ft[i],fr[i],modal[i])
                    if diagnostics:
                        energies[f'body.{name}.strain']=strain[i]
        else:
            nodal = {name: NodalWrench.zeros(body.model.n_node, device=body.model.X.device, dtype=body.model.X.dtype) for name, body in self.system.bodies.items()}
            forces = {name: GeneralizedForce.zeros(body.model.n_mode, device=body.model.X.device, dtype=body.model.X.dtype) for name, body in self.system.bodies.items()}
            for name, body in self.system.bodies.items():
                restoring_force, strain_energy = linear_reduced_force_and_energy(body, diagnostics=diagnostics)
                forces[name].modal.add_(restoring_force)
                if strain_energy is not None:
                    energies[f"body.{name}.strain"] = strain_energy
        interaction_diagnostics: dict[str, dict[str, Any]] = {}
        from FMBD.interaction.single_strand import SingleStrandBeamInteraction
        from FMBD.interaction.tether import SingleStrandTetherInteraction
        linker_results = {}
        if getattr(self.system, 'batch_linkers', True):
            linker_groups = {}
            for index, interaction in enumerate(self.system.interactions):
                if type(interaction) in (SingleStrandBeamInteraction, SingleStrandTetherInteraction):
                    tensor = self.system.bodies[interaction.body_a].model.X
                    key = (type(interaction), interaction.parameters, tensor.device, tensor.dtype)
                    linker_groups.setdefault(key, []).append((index, interaction))
            for entries in linker_groups.values():
                results = type(entries[0][1]).compute_batch(
                    [entry[1] for entry in entries], self.system, context, diagnostics)
                linker_results.update(zip((entry[0] for entry in entries), results))
        for index, interaction in enumerate(self.system.interactions):
            result = linker_results[index] if index in linker_results else interaction.compute(self.system, context, diagnostics=diagnostics)
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
        if batching:
            for group in groups:
                projected=project_nodal_wrenches_batch(group,nodal)
                for name,force in projected.items():
                    forces[name].add_(force)
        else:
            for name, body in self.system.bodies.items():
                forces[name].add_(project_nodal_wrench(body, nodal[name]))
        return forces, energies, interaction_diagnostics

    def step(self, step: int, diagnostics: bool = False) -> StepResult:
        if not self._initialized:
            self.initialize()
        from FMBD.observer.log import SimulationLogWriter
        diagnostics = diagnostics or any(
            isinstance(observer, SimulationLogWriter) and (step + 1) % observer.every == 0
            for observer in self.observers)
        context = self.protocol.context(step)
        forces, energies, interaction_diagnostics = self.evaluate_forces(context, diagnostics)
        integrator_diagnostics = self.integrator.step(
            self.system,
            forces,
            context,
            force_evaluator=lambda midpoint_context: self.evaluate_forces(midpoint_context)[0],
            diagnostics=diagnostics,
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
        if self.output_dir is not None and self._initialized:
            self.save_restart(Path(self.output_dir) / 'restart.pt')

    def save_restart(self, path) -> None:
        from FMBD.restart import save_restart
        save_restart(self, path)

    def load_restart(self, path):
        from FMBD.restart import load_restart
        return load_restart(self, path)

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
                n_steps = self.protocol.n_steps - (self.last_completed_step or 0)
            except AttributeError as exc:
                raise ValueError("n_steps is required for a protocol without n_steps") from exc
        try:
            start_step = self.last_completed_step or 0
            for step in range(start_step, start_step + n_steps):
                diagnostics = diagnostics_every is not None and (step + 1) % diagnostics_every == 0
                result = self.step(step, diagnostics=diagnostics)
                if callback is not None:
                    callback(result)
        finally:
            self.finish()
