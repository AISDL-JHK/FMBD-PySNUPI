"""Overdamped flexible-multibody Brownian dynamics integrator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import torch

from FMBD.core.forces import GeneralizedForce
from FMBD.math.so3 import exp_so3


def _cap_vector(value: torch.Tensor, maximum: float | None) -> torch.Tensor:
    if maximum is None:
        return value
    norm = torch.linalg.norm(value)
    scale = torch.clamp(value.new_tensor(maximum) / norm.clamp_min(1.0e-30), max=1.0)
    return value * scale


@dataclass(frozen=True)
class IntegratorOptions:
    """Controls for the rigid and reduced-coordinate Brownian update.

    ``midpoint_brownian`` reevaluates all forces after an independently
    Brownian-displaced half step. It is the default because it evaluates the
    configuration-dependent drift at the stochastic midpoint. The former
    first-order update remains available as ``euler_brownian``.
    """

    max_translation_step: float | None = None
    max_rotation_step: float | None = None
    max_modal_step: float | None = None
    internal_scheme: str = "nma"
    scheme: str = "midpoint_brownian"

    def __post_init__(self) -> None:
        valid_internal = {"nma", "modal_mean_force", "meanforce_respod_ou"}
        if self.internal_scheme not in valid_internal:
            raise ValueError(
                f"internal_scheme must be one of {sorted(valid_internal)}, got {self.internal_scheme!r}"
            )
        valid_integrators = {"euler_brownian", "midpoint_brownian"}
        if self.scheme not in valid_integrators:
            raise ValueError(
                f"scheme must be one of {sorted(valid_integrators)}, got {self.scheme!r}"
            )


@dataclass
class OverdampedFMBDIntegrator:
    """Body-local rigid mobility plus reduced internal overdamped dynamics.

    The supplied body model owns fixed mobility and noise factors, preserving
    the reference model's full 6x6 translation-rotation coupling.
    """

    dt: float
    options: IntegratorOptions = IntegratorOptions()
    generator: torch.Generator | None = None

    def __post_init__(self) -> None:
        self.dt = float(self.dt)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

    def _modal_force(self, body: Any, force: GeneralizedForce) -> torch.Tensor:
        modal_force = force.modal
        if self.options.internal_scheme != "nma":
            if body.model.modal_mean_force is None:
                raise ValueError(
                    f"{self.options.internal_scheme} requires modal_mean_force in the BodyModel"
                )
            modal_force = modal_force + body.model.modal_mean_force
        return modal_force

    def _validate_ou_interval(self, body: Any) -> None:
        if self.options.internal_scheme != "meanforce_respod_ou":
            return
        model, state = body.model, body.state
        if model.n_respod_mode == 0 or state.a is None:
            raise ValueError("meanforce_respod_ou requires resPOD-OU data and BodyState.a")
        tolerance = 1.0e-12 * max(1.0, abs(float(model.closure_dt_ps)))
        if abs(float(self.dt) - float(model.closure_dt_ps)) > tolerance:
            raise ValueError(
                "FMBD dt and closure_ou_dt_ps differ. Export/fitting and "
                "runtime intervals must be aligned for this scheme."
            )

    def _sample_rigid_noise(self, body: Any, sqrt_dt: torch.Tensor) -> torch.Tensor:
        state, model = body.state, body.model
        if not body.dynamics.pose_brownian:
            return state.q.new_zeros(6)
        xi = torch.randn(
            6, device=state.q.device, dtype=state.q.dtype, generator=self.generator,
        )
        return model.mu_rigid @ (model.S_rigid @ xi * sqrt_dt)

    def _sample_modal_noise(self, body: Any, sqrt_dt: torch.Tensor) -> torch.Tensor:
        state, model = body.state, body.model
        if not body.dynamics.modal_brownian:
            return state.q.new_zeros(model.n_mode)
        xi = torch.randn(
            model.n_mode, device=state.q.device, dtype=state.q.dtype, generator=self.generator,
        )
        return model.Bq @ xi * sqrt_dt

    def _next_ou_state(
        self,
        body: Any,
        rho: torch.Tensor,
        innovation_scale: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.options.internal_scheme != "meanforce_respod_ou":
            return None
        state, model = body.state, body.model
        assert state.a is not None
        if not body.dynamics.modal_brownian:
            return rho * state.a
        eta = torch.randn(
            model.n_respod_mode, device=state.q.device, dtype=state.q.dtype,
            generator=self.generator,
        )
        return rho * state.a + innovation_scale * eta

    def _modal_increment(
        self,
        body: Any,
        force: GeneralizedForce,
        dt: torch.Tensor,
        sqrt_dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Euler update for NMA coordinates and optional residual POD OU modes."""
        state, model = body.state, body.model
        self._validate_ou_interval(body)
        dq = dt * (model.mu @ self._modal_force(body, force))
        dq = dq + self._sample_modal_noise(body, sqrt_dt)

        if self.options.internal_scheme != "meanforce_respod_ou":
            return dq, None
        assert model.respod_ou_rho is not None and model.respod_ou_sigma is not None
        innovation = model.respod_ou_sigma * torch.sqrt(
            (1.0 - model.respod_ou_rho.square()).clamp_min(0.0)
        )
        return dq, self._next_ou_state(body, model.respod_ou_rho, innovation)

    @staticmethod
    def _wrench_in_body_frame(body: Any, force: GeneralizedForce) -> torch.Tensor:
        state = body.state
        return torch.cat((state.R.T @ force.translation, state.R.T @ force.rotation))

    def _apply_full_increment(
        self,
        body: Any,
        *,
        c0: torch.Tensor,
        R0: torch.Tensor,
        q0: torch.Tensor,
        deta_body: torch.Tensor,
        dq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dc_body = _cap_vector(deta_body[:3], self.options.max_translation_step)
        dtheta_body = _cap_vector(deta_body[3:], self.options.max_rotation_step)
        dq = _cap_vector(dq, self.options.max_modal_step)
        dc_global = R0 @ dc_body
        body.state.c = c0 + dc_global
        body.state.R = R0 @ exp_so3(dtheta_body)
        body.state.q = q0 + dq
        return dc_global, dtheta_body, dq

    def _step_euler(
        self,
        system: Any,
        forces: Mapping[str, GeneralizedForce],
    ) -> dict[str, dict[str, torch.Tensor]]:
        diagnostics: dict[str, dict[str, torch.Tensor]] = {}
        for name, body in system.bodies.items():
            if not body.dynamic:
                continue
            force = forces[name]
            state, model = body.state, body.model
            dt = state.q.new_tensor(self.dt)
            sqrt_dt = torch.sqrt(dt)
            deta_body = dt * (model.mu_rigid @ self._wrench_in_body_frame(body, force))
            deta_body = deta_body + self._sample_rigid_noise(body, sqrt_dt)
            dq, next_a = self._modal_increment(body, force, dt, sqrt_dt)
            c0, R0, q0 = state.c, state.R, state.q
            dc_global, dtheta_body, dq = self._apply_full_increment(
                body, c0=c0, R0=R0, q0=q0, deta_body=deta_body, dq=dq,
            )
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

    def _step_midpoint(
        self,
        system: Any,
        forces: Mapping[str, GeneralizedForce],
        context: Any,
        force_evaluator: Callable[[Any], Mapping[str, GeneralizedForce]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Perform one stochastic midpoint Brownian update.

        Independent half-step noises create the midpoint state. Forces are
        evaluated there, then used for a full update from the saved initial
        state. The two noise increments are summed for the final Brownian
        displacement, so the full-step covariance is unchanged.
        """
        saved: dict[str, dict[str, torch.Tensor | None]] = {}
        increments: dict[str, dict[str, torch.Tensor | None]] = {}

        for name, body in system.bodies.items():
            if not body.dynamic:
                continue
            state, model = body.state, body.model
            self._validate_ou_interval(body)
            dt = state.q.new_tensor(self.dt)
            half_dt = 0.5 * dt
            sqrt_half_dt = torch.sqrt(half_dt)
            c0, R0, q0 = state.c.clone(), state.R.clone(), state.q.clone()
            a0 = None if state.a is None else state.a.clone()
            saved[name] = {"c": c0, "R": R0, "q": q0, "a": a0}

            pose_a = self._sample_rigid_noise(body, sqrt_half_dt)
            pose_b = self._sample_rigid_noise(body, sqrt_half_dt)
            modal_a = self._sample_modal_noise(body, sqrt_half_dt)
            modal_b = self._sample_modal_noise(body, sqrt_half_dt)
            half_wrench = self._wrench_in_body_frame(body, forces[name])
            half_deta = half_dt * (model.mu_rigid @ half_wrench) + pose_a
            half_dq = half_dt * (model.mu @ self._modal_force(body, forces[name])) + modal_a

            rho_half = None
            ou_b = None
            if self.options.internal_scheme == "meanforce_respod_ou":
                assert a0 is not None
                assert model.respod_ou_rho is not None and model.respod_ou_sigma is not None
                rho_half = torch.sqrt(model.respod_ou_rho)
                innovation_half = model.respod_ou_sigma * torch.sqrt(
                    (1.0 - rho_half.square()).clamp_min(0.0)
                )
                state.a = self._next_ou_state(body, rho_half, innovation_half)
                # Draw the second innovation before the force callback so one
                # full step consumes a reproducible fixed noise sequence.
                if body.dynamics.modal_brownian:
                    eta_b = torch.randn(
                        model.n_respod_mode, device=state.q.device, dtype=state.q.dtype,
                        generator=self.generator,
                    )
                    ou_b = innovation_half * eta_b
                else:
                    ou_b = state.q.new_zeros(model.n_respod_mode)

            state.c = c0 + R0 @ half_deta[:3]
            state.R = R0 @ exp_so3(half_deta[3:])
            state.q = q0 + half_dq
            increments[name] = {
                "pose_a": pose_a,
                "pose_b": pose_b,
                "modal_a": modal_a,
                "modal_b": modal_b,
                "rho_half": rho_half,
                "ou_b": ou_b,
            }

        try:
            system.reconstruct_all()
            midpoint_forces = force_evaluator(context)
            missing = [
                name for name, body in system.bodies.items()
                if body.dynamic and name not in midpoint_forces
            ]
            if missing:
                raise KeyError(f"midpoint force evaluation omitted dynamic bodies: {missing}")
        except Exception:
            for name, values in saved.items():
                state = system.bodies[name].state
                state.c = values["c"]
                state.R = values["R"]
                state.q = values["q"]
                state.a = values["a"]
            system.reconstruct_all()
            raise

        diagnostics: dict[str, dict[str, torch.Tensor]] = {}
        for name, body in system.bodies.items():
            if not body.dynamic:
                continue
            state, model = body.state, body.model
            values = saved[name]
            noise = increments[name]
            c0, R0, q0 = values["c"], values["R"], values["q"]
            assert isinstance(c0, torch.Tensor) and isinstance(R0, torch.Tensor) and isinstance(q0, torch.Tensor)
            force = midpoint_forces[name]
            dt = state.q.new_tensor(self.dt)
            deta = dt * (model.mu_rigid @ self._wrench_in_body_frame(body, force))
            deta = deta + noise["pose_a"] + noise["pose_b"]
            dq = dt * (model.mu @ self._modal_force(body, force))
            dq = dq + noise["modal_a"] + noise["modal_b"]
            dc_global, dtheta_body, dq = self._apply_full_increment(
                body, c0=c0, R0=R0, q0=q0, deta_body=deta, dq=dq,
            )

            a0 = values["a"]
            da_norm = None
            if self.options.internal_scheme == "meanforce_respod_ou":
                assert isinstance(a0, torch.Tensor)
                assert isinstance(noise["rho_half"], torch.Tensor) and isinstance(noise["ou_b"], torch.Tensor)
                assert state.a is not None
                state.a = noise["rho_half"] * state.a + noise["ou_b"]
                da_norm = torch.linalg.norm(state.a - a0).detach()

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

    def step(
        self,
        system: Any,
        forces: Mapping[str, GeneralizedForce],
        context: Any,
        *,
        force_evaluator: Callable[[Any], Mapping[str, GeneralizedForce]] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Advance one time step.

        ``midpoint_brownian`` requires ``force_evaluator`` because forces must
        be recomputed from the reconstructed midpoint configuration. The
        :class:`~FMBD.Simulation` orchestration supplies it automatically.
        """
        if self.options.scheme == "euler_brownian":
            return self._step_euler(system, forces)
        if force_evaluator is None:
            raise ValueError(
                "midpoint_brownian requires force_evaluator; call through Simulation "
                "or provide a callback that evaluates midpoint forces"
            )
        return self._step_midpoint(system, forces, context, force_evaluator)
