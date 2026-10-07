"""CPU NumPy/SciPy and GPU CuPy overdamped Brownian integrators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from FMBD.legacy_numpy_cupy.backend import cupy_module
from FMBD.legacy_numpy_cupy.core.forces import GeneralizedForce
from FMBD.legacy_numpy_cupy.math.so3 import exp_so3


def _cap_vector(value, maximum):
    if maximum is None:
        return value
    xp = cupy_module() if type(value).__module__.split(".", 1)[0] == "cupy" else np
    norm = xp.linalg.norm(value)
    return value * xp.minimum(1.0, maximum / xp.maximum(norm, 1.0e-30))


@dataclass(frozen=True)
class IntegratorOptions:
    max_translation_step: float | None = None
    max_rotation_step: float | None = None
    max_modal_step: float | None = None
    internal_scheme: str = "nma"
    scheme: str = "midpoint_brownian"

    def __post_init__(self):
        if self.scheme not in {"midpoint_brownian", "euler_brownian"}:
            raise ValueError("scheme must be 'midpoint_brownian' or 'euler_brownian'")
        valid = {"nma", "modal_mean_force", "meanforce_respod_ou"}
        if self.internal_scheme not in valid:
            raise ValueError(f"internal_scheme must be one of {sorted(valid)}")


@dataclass
class OverdampedFMBDIntegrator:
    dt: float
    options: IntegratorOptions = IntegratorOptions()
    seed: int = 0
    _generators: dict[str, object] | None = None

    def __post_init__(self):
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self._generators is None:
            self._generators = {}

    def _generator(self, xp, device: int):
        if xp is np:
            key = "cpu"
            if key not in self._generators:
                self._generators[key] = np.random.default_rng(self.seed)
        else:
            key = f"cuda:{device}"
            if key not in self._generators:
                cp = cupy_module()
                with cp.cuda.Device(device):
                    self._generators[key] = cp.random.RandomState(self.seed)
        return self._generators[key]

    @staticmethod
    def _normal(rng, xp, size, dtype):
        if xp is np:
            return rng.standard_normal(size).astype(dtype, copy=False)
        return rng.standard_normal(size, dtype=dtype)

    def _effective_modal_force(self, body, force):
        scheme = self.options.internal_scheme
        if scheme == "nma":
            return force.modal
        if body.model.modal_mean_force is None:
            raise ValueError(f"{scheme} requires modal_mean_force in the BodyModel")
        return force.modal + body.model.modal_mean_force

    def _check_closure_dt(self, body):
        model = body.model
        if self.options.internal_scheme != "meanforce_respod_ou":
            return
        if model.n_respod_mode == 0 or body.state.a is None:
            raise ValueError("meanforce_respod_ou requires resPOD-OU data and BodyState.a")
        tol = 1.0e-12 * max(1.0, abs(float(model.closure_dt_ps)))
        if abs(self.dt - float(model.closure_dt_ps)) > tol:
            raise ValueError("FMBD dt and closure_ou_dt_ps differ; use the fitted OU interval")

    def _step_midpoint(self, system, forces, context, force_evaluator):
        dt, root_half = self.dt, np.sqrt(self.dt / 2.0)
        base = {}
        mid_noise = {}
        for name, body in system.bodies.items():
            if not body.dynamic:
                continue
            state, model, xp = body.state, body.model, body.model.xp
            self._check_closure_dt(body)
            device = int(model.X.device.id) if model.backend == "cuda" else 0
            rng = self._generator(xp, device)
            base[name] = tuple(None if value is None else value.copy() for value in (state.q, state.R, state.c, state.a))
            force = forces[name]
            wrench = xp.concatenate((state.R.T @ force.translation, state.R.T @ force.rotation))
            pose_noise_a = pose_noise_b = xp.zeros(6, dtype=model.dtype)
            if body.dynamics.pose_brownian:
                xi_a = self._normal(rng, xp, 6, model.dtype)
                xi_b = self._normal(rng, xp, 6, model.dtype)
                pose_noise_a = model.mu_rigid @ (model.S_rigid @ xi_a * root_half)
                pose_noise_b = model.mu_rigid @ (model.S_rigid @ xi_b * root_half)
            modal_noise_a = modal_noise_b = xp.zeros(model.n_mode, dtype=model.dtype)
            if body.dynamics.modal_brownian:
                xi_a = self._normal(rng, xp, model.n_mode, model.dtype)
                xi_b = self._normal(rng, xp, model.n_mode, model.dtype)
                modal_noise_a = model.Bq @ xi_a * root_half
                modal_noise_b = model.Bq @ xi_b * root_half
            deta_half = 0.5 * dt * (model.mu_rigid @ wrench) + pose_noise_a
            dq_half = 0.5 * dt * (model.mu @ self._effective_modal_force(body, force)) + modal_noise_a
            state.c = state.c + state.R @ deta_half[:3]
            state.R = state.R @ exp_so3(deta_half[3:])
            state.q = state.q + dq_half
            a_noise_a = a_noise_b = None
            if self.options.internal_scheme == "meanforce_respod_ou":
                rho_half = xp.sqrt(model.respod_ou_rho)
                amp_half = model.respod_ou_sigma * xp.sqrt(xp.maximum(1.0 - rho_half**2, 0.0))
                if body.dynamics.modal_brownian:
                    eta_a = self._normal(rng, xp, model.n_respod_mode, model.dtype)
                    eta_b = self._normal(rng, xp, model.n_respod_mode, model.dtype)
                    a_noise_a, a_noise_b = amp_half * eta_a, amp_half * eta_b
                else:
                    a_noise_a = a_noise_b = xp.zeros(model.n_respod_mode, dtype=model.dtype)
                state.a = rho_half * state.a + a_noise_a
            mid_noise[name] = (pose_noise_a, pose_noise_b, modal_noise_a, modal_noise_b, a_noise_a, a_noise_b)

        system.reconstruct_all()
        midpoint_forces = force_evaluator(context) if force_evaluator is not None else forces

        diagnostics = {}
        for name, body in system.bodies.items():
            if not body.dynamic:
                continue
            state, model, xp = body.state, body.model, body.model.xp
            q0, R0, c0, a0 = base[name]
            pose_a, pose_b, modal_a, modal_b, a_noise_a, a_noise_b = mid_noise[name]
            force_mid = midpoint_forces[name]
            wrench_mid = xp.concatenate((state.R.T @ force_mid.translation, state.R.T @ force_mid.rotation))
            deta = dt * (model.mu_rigid @ wrench_mid) + pose_a + pose_b
            deta = xp.concatenate((_cap_vector(deta[:3], self.options.max_translation_step), _cap_vector(deta[3:], self.options.max_rotation_step)))
            dq = dt * (model.mu @ self._effective_modal_force(body, force_mid)) + modal_a + modal_b
            dq = _cap_vector(dq, self.options.max_modal_step)
            state.c = c0 + R0 @ deta[:3]
            state.R = R0 @ exp_so3(deta[3:])
            state.q = q0 + dq
            da_norm = None
            if self.options.internal_scheme == "meanforce_respod_ou":
                rho_half = xp.sqrt(model.respod_ou_rho)
                state.a = rho_half * state.a + a_noise_b
                da_norm = xp.linalg.norm(state.a - a0)
            diagnostics[name] = self._diagnostics(xp, state, q0, dq, da_norm, force_mid, deta)
        return diagnostics

    def _step_euler(self, system, forces, context):
        diagnostics = {}
        dt = self.dt
        for name, body in system.bodies.items():
            if not body.dynamic:
                continue
            state, model, xp = body.state, body.model, body.model.xp
            self._check_closure_dt(body)
            device = int(model.X.device.id) if model.backend == "cuda" else 0
            rng = self._generator(xp, device)
            force = forces[name]
            wrench = xp.concatenate((state.R.T @ force.translation, state.R.T @ force.rotation))
            deta = dt * (model.mu_rigid @ wrench)
            if body.dynamics.pose_brownian:
                xi = self._normal(rng, xp, 6, model.dtype)
                deta += model.mu_rigid @ (model.S_rigid @ xi * np.sqrt(dt))
            dq = dt * (model.mu @ self._effective_modal_force(body, force))
            if body.dynamics.modal_brownian:
                dq += model.Bq @ self._normal(rng, xp, model.n_mode, model.dtype) * np.sqrt(dt)
            state.c += state.R @ _cap_vector(deta[:3], self.options.max_translation_step)
            state.R = state.R @ exp_so3(_cap_vector(deta[3:], self.options.max_rotation_step))
            state.q += _cap_vector(dq, self.options.max_modal_step)
            da_norm = None
            if self.options.internal_scheme == "meanforce_respod_ou":
                eta = self._normal(rng, xp, model.n_respod_mode, model.dtype)
                state.a = model.respod_ou_rho * state.a + model.respod_ou_sigma * xp.sqrt(xp.maximum(1.0 - model.respod_ou_rho**2, 0.0)) * eta
                da_norm = xp.linalg.norm(state.a)
            diagnostics[name] = self._diagnostics(xp, state, state.q - dq, dq, da_norm, force, deta)
        return diagnostics

    @staticmethod
    def _diagnostics(xp, state, q0, dq, da_norm, force, deta):
        result = {
            "dc_norm": xp.linalg.norm(deta[:3]), "dw_norm": xp.linalg.norm(deta[3:]),
            "dq_norm": xp.linalg.norm(dq), "Ft_norm": xp.linalg.norm(force.translation),
            "Fr_norm": xp.linalg.norm(force.rotation), "Fq_norm": xp.linalg.norm(force.modal),
        }
        if da_norm is not None:
            result["da_norm"] = da_norm
        return result

    def step(self, system, forces: Mapping[str, GeneralizedForce], context, *, force_evaluator: Callable | None = None):
        if self.options.scheme == "midpoint_brownian":
            return self._step_midpoint(system, forces, context, force_evaluator)
        return self._step_euler(system, forces, context)


__all__ = ["IntegratorOptions", "OverdampedFMBDIntegrator"]
