"""Constitutive laws for deformation internal to a single body."""

from __future__ import annotations

import torch

from FMBD.core.body import Body


def linear_reduced_force(body: Body) -> torch.Tensor:
    """Return the linear reduced elastic restoring force ``-K q``."""
    return -(body.model.K @ body.state.q)


def linear_reduced_energy(body: Body) -> torch.Tensor:
    """Return ``0.5 qᵀKq`` in pN nm."""
    Kq = body.model.K @ body.state.q
    return 0.5 * torch.dot(body.state.q, Kq)


def linear_reduced_force_and_energy(body: Body, *, diagnostics: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute ``-Kq`` once and optionally its associated strain energy."""
    Kq = body.model.K @ body.state.q
    energy = 0.5 * torch.dot(body.state.q, Kq) if diagnostics else None
    return -Kq, energy
