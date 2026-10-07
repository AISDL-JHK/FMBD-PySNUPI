"""NumPy/CuPy reduced body models and mutable runtime state."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from FMBD.legacy_numpy_cupy.backend import ArrayBackend, array_module, resolve_backend, to_numpy
from FMBD.legacy_numpy_cupy.math.so3 import exp_so3_batch

BODY_ROM_SCHEMA = "fmbd.body_rom"
BODY_ROM_VERSION = 1


def _normalize_connectivity(value: Any, n_node: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"connectivity must have shape (E, 2), got {arr.shape}")
    if arr.min() < 0 or arr.max() >= n_node:
        raise IndexError("connectivity contains an out-of-range node index")
    arr = arr[arr[:, 0] != arr[:, 1]]
    return np.unique(np.sort(arr, axis=1), axis=0)


@dataclass(frozen=True)
class BodyModel:
    """Static reduced body data stored as NumPy (CPU) or CuPy (CUDA) arrays."""

    name: str
    X: Any
    Q0: Any
    phi: Any
    phi_pos: Any
    phi_rot: Any
    K: Any
    Gamma: Any
    mu: Any
    S: Any
    Bq: Any
    Z_rigid: Any
    mu_rigid: Any
    S_rigid: Any
    rigid_reference_center: Any
    connectivity: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    modal_mean_force: Any | None = None
    respod_psi: Any | None = None
    respod_ou_rho: Any | None = None
    respod_ou_sigma: Any | None = None
    closure_dt_ps: float | None = None

    @property
    def xp(self):
        return array_module(self.X)

    @property
    def backend(self) -> str:
        return "cuda" if self.xp is not np else "cpu"

    @property
    def device(self) -> str:
        if self.backend == "cpu":
            return "cpu"
        return f"cuda:{int(self.X.device.id)}"

    @property
    def dtype(self):
        return self.X.dtype

    @property
    def n_node(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_mode(self) -> int:
        return int(self.K.shape[0])

    @property
    def n_respod_mode(self) -> int:
        return 0 if self.respod_psi is None else int(self.respod_psi.shape[1])

    def to_dict(self) -> dict[str, Any]:
        names = ("X", "Q0", "phi", "phi_pos", "phi_rot", "K", "Gamma", "mu", "S", "Bq", "Z_rigid", "mu_rigid", "S_rigid", "rigid_reference_center")
        data = {name: to_numpy(getattr(self, name)).copy() for name in names}
        data.update({"schema": BODY_ROM_SCHEMA, "version": BODY_ROM_VERSION, "name": self.name, "connectivity": self.connectivity.copy(), "metadata": dict(self.metadata)})
        for name in ("modal_mean_force", "respod_psi", "respod_ou_rho", "respod_ou_sigma"):
            value = getattr(self, name)
            if value is not None:
                data[name] = to_numpy(value).copy()
        if self.respod_psi is not None:
            data["closure_dt_ps"] = self.closure_dt_ps
        return data

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            pickle.dump(self.to_dict(), stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path, *, backend: str = "cpu", device: int = 0, dtype=np.float64) -> "BodyModel":
        with Path(path).open("rb") as stream:
            return cls.from_dict(pickle.load(stream), backend=backend, device=device, dtype=dtype)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, backend: str = "cpu", device: int = 0, dtype=np.float64) -> "BodyModel":
        if data.get("schema") != BODY_ROM_SCHEMA:
            raise ValueError("not an FMBD BodyModel artifact; use from_fmbd_data for SNUPY MOR data")
        if data.get("version") != BODY_ROM_VERSION:
            raise ValueError(f"unsupported BodyModel version: {data.get('version')!r}")
        xp_backend = resolve_backend(backend, device)
        with xp_backend.context():
            xp = xp_backend.xp
            names = ("X", "Q0", "phi", "phi_pos", "phi_rot", "K", "Gamma", "mu", "S", "Bq", "Z_rigid", "mu_rigid", "S_rigid", "rigid_reference_center")
            arrays = {name: xp.asarray(data[name], dtype=dtype) for name in names}
            optional = {name: None if data.get(name) is None else xp.asarray(data[name], dtype=dtype) for name in ("modal_mean_force", "respod_psi", "respod_ou_rho", "respod_ou_sigma")}
            model = cls(
                name=str(data["name"]), connectivity=_normalize_connectivity(data["connectivity"], arrays["X"].shape[0]),
                metadata=dict(data.get("metadata", {})), closure_dt_ps=data.get("closure_dt_ps"), **arrays, **optional,
            )
        model._validate()
        return model

    @classmethod
    def from_fmbd_data(cls, path: str | Path, *, name: str | None = None, backend: str = "cpu", device: int = 0, dtype=np.float64) -> "BodyModel":
        """Load the stable ``snupy.mor_fmbd_input`` artifact."""
        with Path(path).open("rb") as stream:
            data = pickle.load(stream)
        if data.get("schema") != "snupy.mor_fmbd_input":
            raise ValueError("not a SNUPY FMBD_data artifact")
        node = np.asarray(data["init_node"], dtype=np.float64)
        if node.ndim != 2 or node.shape[1] < 6:
            raise ValueError("FMBD_data init_node must have shape (N, >=6)")
        n = len(node)
        xp_backend = resolve_backend(backend, device)
        with xp_backend.context():
            xp = xp_backend.xp
            arr = lambda v: xp.asarray(v, dtype=dtype)
            phi = arr(data["phi"])
            if phi.shape[0] != 6 * n:
                raise ValueError("FMBD_data phi is incompatible with init_node")
            phi_node = phi.reshape(n, 6, -1)
            K, Gamma = arr(data["K_r"]), arr(data["Gamma_r"])
            K, Gamma = (K + K.T) * 0.5, (Gamma + Gamma.T) * 0.5
            mu = arr(data["mu_r"]) if "mu_r" in data else xp.linalg.pinv(Gamma)
            mu = (mu + mu.T) * 0.5
            residual = bool(data.get("closure_enabled", False))
            model = cls(
                name=name or str(data.get("name", Path(path).stem)),
                X=arr(node[:, :3]), Q0=exp_so3_batch(arr(node[:, 3:6])),
                phi=phi, phi_pos=phi_node[:, :3], phi_rot=phi_node[:, 3:],
                K=K, Gamma=Gamma, mu=mu, S=arr(data["S_r"]), Bq=arr(data.get("Bq", mu @ arr(data["S_r"]))),
                Z_rigid=(arr(data["Z_rigid"]) + arr(data["Z_rigid"]).T) * 0.5,
                mu_rigid=(arr(data["mu_rigid"]) + arr(data["mu_rigid"]).T) * 0.5,
                S_rigid=arr(data["S_rigid"]), rigid_reference_center=arr(data["rigid_reference_center"]),
                connectivity=_normalize_connectivity(data.get("e_conn", np.empty((0, 2))), n),
                metadata={"source_schema": "snupy.mor_fmbd_input", "source_path": str(path), "snupy_topology": data.get("snupy_topology")},
                modal_mean_force=arr(data.get("r_pre_r", np.zeros(phi.shape[1]))),
                respod_psi=arr(data["respod_psi"]) if residual else None,
                respod_ou_rho=arr(data["respod_ou_rho"]) if residual else None,
                respod_ou_sigma=arr(data["respod_ou_sigma"]) if residual else None,
                closure_dt_ps=float(data["closure_ou_dt_ps"]) if residual else None,
            )
        model._validate()
        return model

    @classmethod
    def from_legacy_mor_data(cls, path: str | Path, *, name: str | None = None, backend: str = "cpu", device: int = 0, dtype=np.float64, drop_rigid_modes: bool = True) -> "BodyModel":
        with Path(path).open("rb") as stream:
            data = pickle.load(stream)
        node = np.asarray(data["init_node"], dtype=np.float64)
        if node.ndim != 2 or node.shape[1] < 6:
            raise ValueError("legacy init_node must have at least six columns")
        n = len(node)
        xp_backend = resolve_backend(backend, device)
        with xp_backend.context():
            xp = xp_backend.xp
            arr = lambda v: xp.asarray(v, dtype=dtype)
            phi, K, Gamma, S = arr(data["phi"]), arr(data["K_r"]), arr(data["Gamma_r"]), arr(data["S_r"])
            K, Gamma = (K + K.T) * 0.5, (Gamma + Gamma.T) * 0.5
            mu = arr(data["mu_r"]) if data.get("mu_r") is not None else None
            if drop_rigid_modes and K.shape[0] > 6:
                eig = xp.linalg.eigvalsh(K)
                if float(to_numpy(xp.max(xp.abs(eig[:6])))) < max(1.0e-12 * float(to_numpy(xp.max(xp.abs(eig)))), 1.0e-6 * abs(float(to_numpy(eig[6])))):
                    phi, K, Gamma, S = phi[:, 6:], K[6:, 6:], Gamma[6:, 6:], S[6:, 6:]
                    if mu is not None:
                        mu = mu[6:, 6:]
            if mu is None:
                mu = xp.linalg.pinv(Gamma)
            mu = (mu + mu.T) * 0.5
            phi_node = phi.reshape(n, 6, -1)
            required = ("Z_rigid", "mu_rigid", "S_rigid", "rigid_reference_center")
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"legacy MOR data is missing rigid hydrodynamics: {missing}")
            arrz, arrmu = arr(data["Z_rigid"]), arr(data["mu_rigid"])
            model = cls(
                name=name or Path(path).stem, X=arr(node[:, :3]), Q0=exp_so3_batch(arr(node[:, 3:6])), phi=phi,
                phi_pos=phi_node[:, :3], phi_rot=phi_node[:, 3:], K=K, Gamma=Gamma, mu=mu, S=S, Bq=mu @ S,
                Z_rigid=(arrz + arrz.T) * 0.5, mu_rigid=(arrmu + arrmu.T) * 0.5,
                S_rigid=arr(data["S_rigid"]), rigid_reference_center=arr(data["rigid_reference_center"]),
                connectivity=_normalize_connectivity(data.get("e_conn", np.empty((0, 2))), n),
                metadata={"source_schema": "snupy.legacy_mor", "source_path": str(path)},
            )
        model._validate()
        return model

    def _validate(self) -> None:
        n, m = self.n_node, self.n_mode
        expected = {"X": (n, 3), "Q0": (n, 3, 3), "phi": (6*n, m), "phi_pos": (n, 3, m), "phi_rot": (n, 3, m), "K": (m, m), "Gamma": (m, m), "mu": (m, m), "Z_rigid": (6, 6), "mu_rigid": (6, 6), "S_rigid": (6, 6), "rigid_reference_center": (3,)}
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}")
        if self.S.ndim != 2 or self.S.shape[0] != m or self.Bq.shape != (m, self.S.shape[1]):
            raise ValueError("S and Bq must have shapes (m, k) and (m, k)")
        if self.modal_mean_force is not None and self.modal_mean_force.shape != (m,):
            raise ValueError("modal_mean_force must have shape (n_mode,)")
        residual = (self.respod_psi, self.respod_ou_rho, self.respod_ou_sigma)
        if any(value is not None for value in residual):
            if any(value is None for value in residual) or self.closure_dt_ps is None:
                raise ValueError("resPOD closure requires psi, rho, sigma, and closure_dt_ps")
            p = self.n_respod_mode
            if self.respod_psi.shape != (6*n, p) or self.respod_ou_rho.shape != (p,) or self.respod_ou_sigma.shape != (p,) or self.closure_dt_ps <= 0:
                raise ValueError("resPOD closure dimensions or time step are invalid")
        _normalize_connectivity(self.connectivity, n)


@dataclass
class BodyState:
    q: Any
    R: Any
    c: Any
    a: Any | None = None

    @classmethod
    def at_reference(cls, model: BodyModel, *, R=None, c=None) -> "BodyState":
        xp = model.xp
        return cls(q=xp.zeros(model.n_mode, dtype=model.dtype), R=xp.eye(3, dtype=model.dtype) if R is None else xp.asarray(R, dtype=model.dtype), c=xp.zeros(3, dtype=model.dtype) if c is None else xp.asarray(c, dtype=model.dtype), a=None if model.n_respod_mode == 0 else xp.zeros(model.n_respod_mode, dtype=model.dtype))


@dataclass
class DynamicsOptions:
    pose_brownian: bool = True
    modal_brownian: bool = True


@dataclass
class Body:
    model: BodyModel
    state: BodyState
    reconstruction: Any | None = None
    dynamic: bool = True
    dynamics: DynamicsOptions = field(default_factory=DynamicsOptions)


__all__ = ["Body", "BodyModel", "BodyState", "DynamicsOptions"]
