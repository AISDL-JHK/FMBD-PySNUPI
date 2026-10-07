"""Immutable reduced body models and mutable runtime body state."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from FMBD.legacy_torch.math.so3 import exp_so3_batch


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
    """Static reduced-order description of one flexible body.

    Tensor units are nm, pN, ps, pN nm and rad.  The model deliberately holds
    no pose or modal state, so it can be shared by multiple runtime bodies.
    """

    name: str
    X: torch.Tensor
    Q0: torch.Tensor
    phi: torch.Tensor
    phi_pos: torch.Tensor
    phi_rot: torch.Tensor
    K: torch.Tensor
    Gamma: torch.Tensor
    mu: torch.Tensor
    S: torch.Tensor
    Bq: torch.Tensor
    Z_rigid: torch.Tensor
    mu_rigid: torch.Tensor
    S_rigid: torch.Tensor
    rigid_reference_center: torch.Tensor
    connectivity: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional equilibrium closure exported by SNUPY's FMBD_data.pkl.
    # ``modal_mean_force`` shifts the NMA equilibrium; ``respod_*`` describe
    # zero-mean local residual fluctuations and are intentionally immutable.
    modal_mean_force: torch.Tensor | None = None
    respod_psi: torch.Tensor | None = None
    respod_ou_rho: torch.Tensor | None = None
    respod_ou_sigma: torch.Tensor | None = None
    closure_dt_ps: float | None = None

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
        """Return a portable, versioned dictionary—not a pickled class instance."""
        tensor_names = (
            "X", "Q0", "phi", "phi_pos", "phi_rot", "K", "Gamma", "mu", "S", "Bq",
            "Z_rigid", "mu_rigid", "S_rigid", "rigid_reference_center",
        )
        data = {name: getattr(self, name).detach().cpu().numpy().copy() for name in tensor_names}
        data.update({
            "schema": BODY_ROM_SCHEMA,
            "version": BODY_ROM_VERSION,
            "name": self.name,
            "connectivity": self.connectivity.copy(),
            "metadata": dict(self.metadata),
        })
        if self.modal_mean_force is not None:
            data["modal_mean_force"] = self.modal_mean_force.detach().cpu().numpy().copy()
        if self.respod_psi is not None:
            data["respod_psi"] = self.respod_psi.detach().cpu().numpy().copy()
            data["respod_ou_rho"] = self.respod_ou_rho.detach().cpu().numpy().copy()
            data["respod_ou_sigma"] = self.respod_ou_sigma.detach().cpu().numpy().copy()
            data["closure_dt_ps"] = self.closure_dt_ps
        return data

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as fh:
            pickle.dump(self.to_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64) -> "BodyModel":
        with Path(path).open("rb") as fh:
            data = pickle.load(fh)
        return cls.from_dict(data, device=device, dtype=dtype)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64) -> "BodyModel":
        if data.get("schema") != BODY_ROM_SCHEMA:
            raise ValueError("not an FMBD BodyModel artifact; use from_legacy_mor_data for SNUPY MOR data")
        if data.get("version") != BODY_ROM_VERSION:
            raise ValueError(f"unsupported BodyModel version: {data.get('version')!r}")
        dev = torch.device(device)
        required = ("name", "X", "Q0", "phi", "phi_pos", "phi_rot", "K", "Gamma", "mu", "S", "Bq", "Z_rigid", "mu_rigid", "S_rigid", "rigid_reference_center", "connectivity")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"BodyModel artifact is missing fields: {missing}")
        tensor_names = required[1:-1]
        tensors = {name: torch.as_tensor(data[name], device=dev, dtype=dtype) for name in tensor_names}
        optional = {
            "modal_mean_force": None if data.get("modal_mean_force") is None else torch.as_tensor(data["modal_mean_force"], device=dev, dtype=dtype),
            "respod_psi": None if data.get("respod_psi") is None else torch.as_tensor(data["respod_psi"], device=dev, dtype=dtype),
            "respod_ou_rho": None if data.get("respod_ou_rho") is None else torch.as_tensor(data["respod_ou_rho"], device=dev, dtype=dtype),
            "respod_ou_sigma": None if data.get("respod_ou_sigma") is None else torch.as_tensor(data["respod_ou_sigma"], device=dev, dtype=dtype),
            "closure_dt_ps": data.get("closure_dt_ps"),
        }
        model = cls(name=str(data["name"]), connectivity=np.asarray(data["connectivity"], dtype=np.int64), metadata=dict(data.get("metadata", {})), **tensors, **optional)
        model._validate()
        return model

    @classmethod
    def from_fmbd_data(cls, path: str | Path, *, name: str | None = None, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64) -> "BodyModel":
        """Load a SNUPY-prepared ``FMBD_data.pkl`` artifact.

        This adapter is the boundary between SNUPY offline preparation and the
        independent FMBD runtime. It contains no SNUPY imports.
        """
        with Path(path).open("rb") as fh:
            data = pickle.load(fh)
        if data.get("schema") != "snupy.mor_fmbd_input":
            raise ValueError("not a SNUPY FMBD_data artifact")
        dev = torch.device(device)
        init_node = np.asarray(data["init_node"], dtype=np.float64)
        if init_node.ndim != 2 or init_node.shape[1] < 6:
            raise ValueError("FMBD_data init_node must have shape (N, >=6)")
        n_node = init_node.shape[0]
        phi = torch.as_tensor(data["phi"], device=dev, dtype=dtype)
        if phi.shape[0] != 6 * n_node:
            raise ValueError("FMBD_data phi is incompatible with init_node")
        phi_node = phi.reshape(n_node, 6, -1)
        K = torch.as_tensor(data["K_r"], device=dev, dtype=dtype)
        Gamma = torch.as_tensor(data["Gamma_r"], device=dev, dtype=dtype)
        K = 0.5 * (K + K.T)
        Gamma = 0.5 * (Gamma + Gamma.T)
        mu = torch.as_tensor(data.get("mu_r", np.linalg.pinv(np.asarray(data["Gamma_r"]))), device=dev, dtype=dtype)
        mu = 0.5 * (mu + mu.T)
        S = torch.as_tensor(data["S_r"], device=dev, dtype=dtype)
        q0 = torch.as_tensor(init_node[:, 3:6], device=dev, dtype=dtype)
        optional = {
            "modal_mean_force": torch.as_tensor(data.get("r_pre_r", np.zeros(phi.shape[1])), device=dev, dtype=dtype),
            "respod_psi": None,
            "respod_ou_rho": None,
            "respod_ou_sigma": None,
            "closure_dt_ps": None,
        }
        if data.get("closure_enabled", False):
            optional.update({
                "respod_psi": torch.as_tensor(data["respod_psi"], device=dev, dtype=dtype),
                "respod_ou_rho": torch.as_tensor(data["respod_ou_rho"], device=dev, dtype=dtype),
                "respod_ou_sigma": torch.as_tensor(data["respod_ou_sigma"], device=dev, dtype=dtype),
                "closure_dt_ps": float(data["closure_ou_dt_ps"]),
            })
        model = cls(
            name=name or str(data.get("name", Path(path).stem)),
            X=torch.as_tensor(init_node[:, :3], device=dev, dtype=dtype),
            Q0=exp_so3_batch(q0),
            phi=phi, phi_pos=phi_node[:, :3, :], phi_rot=phi_node[:, 3:, :],
            K=K, Gamma=Gamma, mu=mu, S=S,
            Bq=torch.as_tensor(data.get("Bq", mu @ S), device=dev, dtype=dtype),
            Z_rigid=0.5 * (torch.as_tensor(data["Z_rigid"], device=dev, dtype=dtype) + torch.as_tensor(data["Z_rigid"], device=dev, dtype=dtype).T),
            mu_rigid=0.5 * (torch.as_tensor(data["mu_rigid"], device=dev, dtype=dtype) + torch.as_tensor(data["mu_rigid"], device=dev, dtype=dtype).T),
            S_rigid=torch.as_tensor(data["S_rigid"], device=dev, dtype=dtype),
            rigid_reference_center=torch.as_tensor(data["rigid_reference_center"], device=dev, dtype=dtype),
            connectivity=_normalize_connectivity(data.get("e_conn", np.empty((0, 2))), n_node),
            metadata={
                "source_schema": "snupy.mor_fmbd_input",
                "source_path": str(path),
                # Optional node-level topology exported by SNUPY.  Keeping it
                # as immutable model metadata lets observers reproduce the
                # original NN/ET/DO PDB convention without touching dynamics.
                "snupy_topology": data.get("snupy_topology"),
            },
            **optional,
        )
        model._validate()
        return model

    @classmethod
    def from_legacy_mor_data(cls, path: str | Path, *, name: str | None = None, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64, drop_rigid_modes: bool = True) -> "BodyModel":
        """Convert a legacy SNUPY MOR dictionary into this runtime model.

        This compatibility adapter is intentionally isolated from the FMBD
        simulation loop and should eventually be replaced by offline export.
        """
        with Path(path).open("rb") as fh:
            data = pickle.load(fh)
        dev = torch.device(device)
        init_node = np.asarray(data["init_node"], dtype=np.float64)
        if init_node.ndim != 2 or init_node.shape[1] < 6:
            raise ValueError("legacy init_node must have at least six columns")
        n_node = init_node.shape[0]
        phi = torch.as_tensor(data["phi"], device=dev, dtype=dtype)
        K = 0.5 * (torch.as_tensor(data["K_r"], device=dev, dtype=dtype) + torch.as_tensor(data["K_r"], device=dev, dtype=dtype).T)
        Gamma = 0.5 * (torch.as_tensor(data["Gamma_r"], device=dev, dtype=dtype) + torch.as_tensor(data["Gamma_r"], device=dev, dtype=dtype).T)
        S = torch.as_tensor(data["S_r"], device=dev, dtype=dtype)
        mu_value = data.get("mu_r")
        mu = None if mu_value is None else torch.as_tensor(mu_value, device=dev, dtype=dtype)
        if drop_rigid_modes and K.shape[0] > 6:
            evals = torch.linalg.eigvalsh(K)
            if torch.max(torch.abs(evals[:6])) < max(1.0e-12 * torch.max(torch.abs(evals)).item(), 1.0e-6 * abs(evals[6].item())):
                phi, K, Gamma, S = phi[:, 6:], K[6:, 6:], Gamma[6:, 6:], S[6:, 6:]
                if mu is not None:
                    mu = mu[6:, 6:]
        if mu is None:
            mu = torch.linalg.pinv(Gamma)
        mu = 0.5 * (mu + mu.T)
        phi_node = phi.reshape(n_node, 6, -1)
        required_rigid = ("Z_rigid", "mu_rigid", "S_rigid", "rigid_reference_center")
        missing = [key for key in required_rigid if key not in data]
        if missing:
            raise KeyError(f"legacy MOR data is missing rigid hydrodynamics: {missing}")
        model = cls(
            name=name or Path(path).stem,
            X=torch.as_tensor(init_node[:, :3], device=dev, dtype=dtype),
            Q0=exp_so3_batch(torch.as_tensor(init_node[:, 3:6], device=dev, dtype=dtype)),
            phi=phi,
            phi_pos=phi_node[:, :3, :],
            phi_rot=phi_node[:, 3:6, :],
            K=K,
            Gamma=Gamma,
            mu=mu,
            S=S,
            Bq=mu @ S,
            Z_rigid=0.5 * (torch.as_tensor(data["Z_rigid"], device=dev, dtype=dtype) + torch.as_tensor(data["Z_rigid"], device=dev, dtype=dtype).T),
            mu_rigid=0.5 * (torch.as_tensor(data["mu_rigid"], device=dev, dtype=dtype) + torch.as_tensor(data["mu_rigid"], device=dev, dtype=dtype).T),
            S_rigid=torch.as_tensor(data["S_rigid"], device=dev, dtype=dtype),
            rigid_reference_center=torch.as_tensor(data["rigid_reference_center"], device=dev, dtype=dtype),
            connectivity=_normalize_connectivity(data.get("e_conn", np.empty((0, 2))), n_node),
            metadata={"source_schema": "snupy.legacy_mor", "source_path": str(path)},
        )
        model._validate()
        return model

    def _validate(self) -> None:
        n, m = self.n_node, self.n_mode
        expected = {"X": (n, 3), "Q0": (n, 3, 3), "phi": (6 * n, m), "phi_pos": (n, 3, m), "phi_rot": (n, 3, m), "K": (m, m), "Gamma": (m, m), "mu": (m, m), "Z_rigid": (6, 6), "mu_rigid": (6, 6), "S_rigid": (6, 6), "rigid_reference_center": (3,)}
        for field_name, shape in expected.items():
            if tuple(getattr(self, field_name).shape) != shape:
                raise ValueError(f"{field_name} must have shape {shape}, got {tuple(getattr(self, field_name).shape)}")
        if self.S.ndim != 2 or self.S.shape[0] != m or self.Bq.shape != (m, self.S.shape[1]):
            raise ValueError("S and Bq must have shapes (m, k) and (m, k)")
        if self.modal_mean_force is not None and tuple(self.modal_mean_force.shape) != (m,):
            raise ValueError("modal_mean_force must have shape (n_mode,)")
        residual = (self.respod_psi, self.respod_ou_rho, self.respod_ou_sigma)
        if any(value is not None for value in residual):
            if any(value is None for value in residual) or self.closure_dt_ps is None:
                raise ValueError("resPOD closure requires psi, rho, sigma, and closure_dt_ps")
            p = self.n_respod_mode
            if tuple(self.respod_psi.shape) != (6 * n, p) or tuple(self.respod_ou_rho.shape) != (p,) or tuple(self.respod_ou_sigma.shape) != (p,):
                raise ValueError("resPOD closure tensor shapes are incompatible")
            if self.closure_dt_ps <= 0:
                raise ValueError("closure_dt_ps must be positive")
        _normalize_connectivity(self.connectivity, n)


@dataclass
class BodyState:
    q: torch.Tensor
    R: torch.Tensor
    c: torch.Tensor
    a: torch.Tensor | None = None

    @classmethod
    def at_reference(cls, model: BodyModel, *, R: torch.Tensor | None = None, c: torch.Tensor | None = None) -> "BodyState":
        return cls(
            q=model.X.new_zeros(model.n_mode),
            R=torch.eye(3, device=model.X.device, dtype=model.X.dtype) if R is None else R,
            c=model.X.new_zeros(3) if c is None else c,
            a=None if model.n_respod_mode == 0 else model.X.new_zeros(model.n_respod_mode),
        )


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
