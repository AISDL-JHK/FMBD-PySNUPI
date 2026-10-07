"""Overdamped flexible-multibody Brownian dynamics integrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from FMBD.legacy_torch.core.forces import GeneralizedForce
from FMBD.legacy_torch.math.so3 import exp_so3


def _cap_vector(value: torch.Tensor, maximum: float | None) -> torch.Tensor:
    if maximum is None:
        return value
    norm = torch.linalg.norm(value)
    scale = torch.clamp(value.new_tensor(maximum) / norm.clamp_min(1.0e-30), max=1.0)
    return value * scale


@dataclass(frozen=True)
class IntegratorOptions:
    max_translation_step: float | None = None
    max_rotation_step: float | None = None
    max_modal_step: float | None = None
    internal_scheme: str = "nma"

    def __post_init__(self) -> None:
        valid = {"nma", "modal_mean_force", "meanforce_respod_ou"}
        if self.internal_scheme not in valid:
            raise ValueError(f"internal_scheme must be one of {sorted(valid)}, got {self.internal_scheme!r}")


@dataclass
class OverdampedFMBDIntegrator:
    """Body-local rigid mobility plus reduced internal overdamped dynamics.

    The supplied body model owns the fixed mobility/noise factors.  This
    preserves the reference model's full 6x6 translation-rotation coupling.
    """

    dt: float
    options: IntegratorOptions = IntegratorOptions()
    generator: torch.Generator | None = None

    def _modal_increment(self, body, force: GeneralizedForce, dt: torch.Tensor, sqrt_dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Integrate NMA coordinates and, optionally, residual POD OU modes."""
        state, model = body.state, body.model
        scheme = self.options.internal_scheme
        modal_force = force.modal
        if scheme != "nma":
            if model.modal_mean_force is None:
                raise ValueError(f"{scheme} requires modal_mean_force in the BodyModel")
            modal_force = modal_force + model.modal_mean_force
        dq = dt * (model.mu @ modal_force)
        if body.dynamics.modal_brownian:
            xi = torch.randn(model.n_mode, device=state.q.device, dtype=state.q.dtype, generator=self.generator)
            dq = dq + model.Bq @ xi * sqrt_dt

        da = None
        if scheme == "meanforce_respod_ou":
            if model.n_respod_mode == 0 or state.a is None:
                raise ValueError("meanforce_respod_ou requires resPOD-OU data and BodyState.a")
            # The offline exporter records rho at the same interval as this
            # integrator's dt. Refuse silent use at an incompatible interval.
            tolerance = 1.0e-12 * max(1.0, abs(float(model.closure_dt_ps)))
            if abs(float(dt.detach().cpu()) - float(model.closure_dt_ps)) > tolerance:
                raise ValueError(
                    "FMBD dt and closure_ou_dt_ps differ. Export/fitting and "
                    "runtime intervals must be aligned for this scheme."
                )
            rho, sigma = model.respod_ou_rho, model.respod_ou_sigma
            if body.dynamics.modal_brownian:
                eta = torch.randn(model.n_respod_mode, device=state.q.device, dtype=state.q.dtype, generator=self.generator)
                da = rho * state.a + sigma * torch.sqrt((1.0 - rho.square()).clamp_min(0.0)) * eta
            else:
                da = rho * state.a
        return dq, da

    def step(self, system, forces: Mapping[str, GeneralizedForce], context) -> dict[str, dict[str, torch.Tensor]]:
        diagnostics: dict[str, dict[str, torch.Tensor]] = {}
        for name, body in system.bodies.items():
            if not body.dynamic:
                continue
            force = forces[name]
            state, model = body.state, body.model
            dt = state.q.new_tensor(self.dt)
            sqrt_dt = torch.sqrt(dt)
            wrench_body = torch.cat((state.R.T @ force.translation, state.R.T @ force.rotation))
            deta_body = dt * (model.mu_rigid @ wrench_body)
            if body.dynamics.pose_brownian:
                xi = torch.randn(6, device=state.q.device, dtype=state.q.dtype, generator=self.generator)
                deta_body = deta_body + model.mu_rigid @ (model.S_rigid @ xi * sqrt_dt)
            dq, next_a = self._modal_increment(body, force, dt, sqrt_dt)
            dc_body = _cap_vector(deta_body[:3], self.options.max_translation_step)
            dtheta_body = _cap_vector(deta_body[3:], self.options.max_rotation_step)
            dq = _cap_vector(dq, self.options.max_modal_step)
            dc_global = state.R @ dc_body
            state.c = state.c + dc_global
            state.R = state.R @ exp_so3(dtheta_body)
            state.q = state.q + dq
            da_norm = None if next_a is None else torch.linalg.norm(next_a - state.a).detach()
            if next_a is not None:
                state.a = next_a
            diagnostics[name] = {
                "dc_norm": torch.linalg.norm(dc_global).detach(),
                "dw_norm": torch.linalg.norm(dtheta_body).detach(),
                "dq_norm": torch.linalg.norm(dq).detach(),
                "Ft_norm": torch.linalg.norm(force.translation).detach(),
                "Fr_norm": torch.linalg.norm(force.rotation).detach(),
                "Fq_norm": torch.linalg.norm(force.modal).detach(),
            }
            if da_norm is not None:
                diagnostics[name]["da_norm"] = da_norm
        return diagnostics
