"""Array backend selection for NumPy CPU and optional CuPy CUDA execution."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np


def cupy_module():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is required for backend='cuda'; install a matching cupy package") from exc
    return cp


def is_cupy_array(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "cupy"


def array_module(*values: Any):
    return cupy_module() if any(is_cupy_array(value) for value in values if value is not None) else np


def to_numpy(value: Any) -> np.ndarray:
    if is_cupy_array(value):
        return cupy_module().asnumpy(value)
    return np.asarray(value)


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    device: int = 0

    def context(self):
        return self.xp.cuda.Device(self.device) if self.name == "cuda" else nullcontext()

    def asarray(self, value, *, dtype=None):
        return self.xp.asarray(value, dtype=dtype)


def resolve_backend(backend: str = "cpu", device: int = 0) -> ArrayBackend:
    name = str(backend).lower()
    if name in {"numpy", "scipy"}:
        name = "cpu"
    if name in {"cupy", "gpu"}:
        name = "cuda"
    if name == "auto":
        name = "cuda" if device is not None and cupy_module().cuda.is_available() else "cpu"
    if name not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu', 'cuda', or 'auto'")
    if isinstance(device, bool) or int(device) != device or int(device) < 0:
        raise ValueError("device must be a non-negative integer")
    if name == "cpu":
        return ArrayBackend("cpu", np, int(device))
    cp = cupy_module()
    if not cp.cuda.is_available():
        raise RuntimeError("backend='cuda' requested but no usable CUDA device is available")
    if int(device) >= cp.cuda.runtime.getDeviceCount():
        raise ValueError(f"CUDA device {device} does not exist")
    return ArrayBackend("cuda", cp, int(device))


__all__ = ["ArrayBackend", "array_module", "cupy_module", "is_cupy_array", "resolve_backend", "to_numpy"]
