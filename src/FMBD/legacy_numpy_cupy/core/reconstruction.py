"""MOR geometry reconstruction shared by all interactions."""

from __future__ import annotations

from dataclasses import dataclass

from FMBD.legacy_numpy_cupy.core.body import Body
from FMBD.legacy_numpy_cupy.math.so3 import exp_so3_batch


@dataclass
class ReconstructedBody:
    U: object
    y: object
    xrel: object
    x: object
    Qlocal: object
    Qglobal: object


def reconstruct(body: Body) -> ReconstructedBody:
    model, state, xp = body.model, body.state, body.model.xp
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
    Qlocal = exp_so3_batch(U[:, 3:]) @ model.Q0
    Qglobal = state.R[None] @ Qlocal
    result = ReconstructedBody(U, y, xrel, x, Qlocal, Qglobal)
    body.reconstruction = result
    return result


__all__ = ["ReconstructedBody", "reconstruct"]
