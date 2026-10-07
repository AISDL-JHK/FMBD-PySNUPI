"""SO(3) maps used by FMBD.

All vectors are rotation vectors in radians.  The implementations intentionally
match the reference switch simulation's Rodrigues convention.
"""

from __future__ import annotations

import torch


def skew(v: torch.Tensor) -> torch.Tensor:
    """Return the 3x3 cross-product matrix for a shape-(3,) vector."""
    if v.shape != (3,):
        raise ValueError(f"expected a shape-(3,) vector, got {tuple(v.shape)}")
    z = v.new_zeros(())
    return torch.stack(
        (
            torch.stack((z, -v[2], v[1])),
            torch.stack((v[2], z, -v[0])),
            torch.stack((-v[1], v[0], z)),
        )
    )


def exp_so3(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map for one rotation vector."""
    theta2 = torch.dot(v, v)
    theta = torch.sqrt(torch.clamp(theta2, min=1.0e-30))
    K = skew(v)
    I = torch.eye(3, device=v.device, dtype=v.dtype)
    a_regular = torch.sin(theta) / theta
    b_regular = (1.0 - torch.cos(theta)) / theta2.clamp_min(1.0e-30)
    a_series = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
    b_series = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    small = theta2 < 1.0e-12
    a = torch.where(small, a_series, a_regular)
    b = torch.where(small, b_series, b_regular)
    return I + a * K + b * (K @ K)


def exp_so3_batch(v: torch.Tensor) -> torch.Tensor:
    """Vectorized Rodrigues map for a tensor with shape ``(N, 3)``."""
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {tuple(v.shape)}")
    n = v.shape[0]
    theta2 = torch.sum(v * v, dim=1)
    theta = torch.sqrt(torch.clamp(theta2, min=1.0e-30))
    K = v.new_zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    I = torch.eye(3, device=v.device, dtype=v.dtype).expand(n, 3, 3)
    a_regular = torch.sin(theta) / theta
    b_regular = (1.0 - torch.cos(theta)) / theta2.clamp_min(1.0e-30)
    a_series = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
    b_series = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    a = torch.where(theta2 < 1.0e-12, a_series, a_regular)[:, None, None]
    b = torch.where(theta2 < 1.0e-12, b_series, b_regular)[:, None, None]
    return I + a * K + b * torch.bmm(K, K)


def log_so3(R: torch.Tensor) -> torch.Tensor:
    """Principal logarithm map for a 3x3 rotation matrix."""
    if R.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got {tuple(R.shape)}")
    cos_theta = ((torch.trace(R) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]))
    regular = 0.5 * theta / torch.sin(theta).clamp_min(1.0e-12) * vee
    return torch.where(theta.abs() < 1.0e-7, 0.5 * vee, regular)
