"""MOR geometry reconstruction shared by all interactions in a step."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from FMBD.core.body import Body
from FMBD.math.so3 import exp_so3_batch


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


def reconstruct_batch(group):
    """Reconstruct different equal-sized models without padding."""
    model=group.bodies[0].model
    for body in group.bodies:
        if body.state.q.shape!=(model.n_mode,) or body.state.R.shape!=(3,3) or body.state.c.shape!=(3,):
            raise ValueError('BodyState shapes must match the model and rigid pose')
        if model.n_respod_mode and (body.state.a is None or body.state.a.shape!=(model.n_respod_mode,)):
            raise ValueError('BodyState.a must match the residual POD mode count')
        if not model.n_respod_mode and body.state.a is not None:
            raise ValueError('BodyState.a is set for a model without POD modes')
    U=group.apply('phi',group.state('q'))
    if model.n_respod_mode:
        U=U+group.apply('respod_psi',group.state('a'))
    U=U.reshape(len(group.bodies),model.n_node,6)
    y=group.model_vectors('X')+U[:,:,:3]
    R=group.state('R')
    xrel=torch.bmm(y,R.transpose(-1,-2))
    x=xrel+group.state('c')[:,None]
    local=exp_so3_batch(U[:,:,3:].reshape(-1,3)).reshape(len(group.bodies),model.n_node,3,3)
    Qlocal=local@group.model_vectors('Q0')
    Qglobal=R[:,None]@Qlocal
    for index,body in enumerate(group.bodies):
        body.reconstruction=ReconstructedBody(U[index],y[index],xrel[index],x[index],Qlocal[index],Qglobal[index])
