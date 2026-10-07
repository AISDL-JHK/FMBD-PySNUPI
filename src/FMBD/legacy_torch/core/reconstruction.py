"""MOR geometry reconstruction shared by all interactions in a step."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from FMBD.legacy_torch.core.body import Body
from FMBD.legacy_torch.math.so3 import exp_so3_batch


@dataclass
class ReconstructedBody:
    U: torch.Tensor
    y: torch.Tensor
    xrel: torch.Tensor
    x: torch.Tensor
    Qlocal: torch.Tensor
    Qglobal: torch.Tensor


def reconstruct(body: Body) -> ReconstructedBody:
    """Reconstruct one body once from its reduced state and cache the result."""
    model, state = body.model, body.state
    if state.q.shape != (model.n_mode,) or state.R.shape != (3, 3) or state.c.shape != (3,):
        raise ValueError("BodyState shapes must be (n_mode,), (3,3), and (3,)")
    U = (model.phi @ state.q).reshape(model.n_node, 6)
    if model.n_respod_mode:
        if state.a is None or state.a.shape != (model.n_respod_mode,):
            raise ValueError("BodyState.a must have shape (n_respod_mode,) for a resPOD body")
        U = U + (model.respod_psi @ state.a).reshape(model.n_node, 6)
    elif state.a is not None:
        raise ValueError("BodyState.a is set but the BodyModel has no resPOD basis")
    y = model.X + U[:, :3]
    xrel = y @ state.R.T
    x = xrel + state.c
    Qlocal = torch.bmm(exp_so3_batch(U[:, 3:6]), model.Q0)
    Qglobal = torch.einsum("ij,njk->nik", state.R, Qlocal)
    result = ReconstructedBody(U=U, y=y, xrel=xrel, x=x, Qlocal=Qlocal, Qglobal=Qglobal)
    body.reconstruction = result
    return result
