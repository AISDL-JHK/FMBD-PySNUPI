"""NumPy/CuPy SO(3) maps used by FMBD."""

from __future__ import annotations

from FMBD.backend import array_module


def skew(v):
    xp = array_module(v)
    value = xp.asarray(v)
    if value.shape != (3,):
        raise ValueError(f"expected a shape-(3,) vector, got {value.shape}")
    x, y, z = value
    return xp.stack((xp.stack((0.0 * x, -z, y)), xp.stack((z, 0.0 * x, -x)), xp.stack((-y, x, 0.0 * x))))


def exp_so3(v):
    xp = array_module(v)
    value = xp.asarray(v)
    if value.shape != (3,):
        raise ValueError(f"expected a shape-(3,) vector, got {value.shape}")
    theta2 = xp.dot(value, value)
    theta = xp.sqrt(xp.maximum(theta2, 1.0e-30))
    K = skew(value)
    identity = xp.eye(3, dtype=value.dtype)
    a = xp.where(theta2 < 1.0e-12, 1.0 - theta2 / 6.0 + theta2**2 / 120.0, xp.sin(theta) / theta)
    b = xp.where(theta2 < 1.0e-12, 0.5 - theta2 / 24.0 + theta2**2 / 720.0, (1.0 - xp.cos(theta)) / xp.maximum(theta2, 1.0e-30))
    return identity + a * K + b * (K @ K)


def exp_so3_batch(v):
    xp = array_module(v)
    value = xp.asarray(v)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {value.shape}")
    theta2 = xp.sum(value * value, axis=1)
    theta = xp.sqrt(xp.maximum(theta2, 1.0e-30))
    K = xp.zeros((len(value), 3, 3), dtype=value.dtype)
    K[:, 0, 1], K[:, 0, 2] = -value[:, 2], value[:, 1]
    K[:, 1, 0], K[:, 1, 2] = value[:, 2], -value[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -value[:, 1], value[:, 0]
    identity = xp.broadcast_to(xp.eye(3, dtype=value.dtype), (len(value), 3, 3))
    a = xp.where(theta2 < 1.0e-12, 1.0 - theta2 / 6.0 + theta2**2 / 120.0, xp.sin(theta) / theta)
    b = xp.where(theta2 < 1.0e-12, 0.5 - theta2 / 24.0 + theta2**2 / 720.0, (1.0 - xp.cos(theta)) / xp.maximum(theta2, 1.0e-30))
    return identity + a[:, None, None] * K + b[:, None, None] * (K @ K)


def log_so3(R):
    xp = array_module(R)
    matrix = xp.asarray(R)
    if matrix.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got {matrix.shape}")
    cosine = xp.clip((xp.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    theta = xp.arccos(cosine)
    vee = xp.stack((matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]))
    regular = 0.5 * theta / xp.maximum(xp.abs(xp.sin(theta)), 1.0e-12) * vee
    return xp.where(xp.abs(theta) < 1.0e-7, 0.5 * vee, regular)


__all__ = ["exp_so3", "exp_so3_batch", "log_so3", "skew"]
