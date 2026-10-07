"""Projection from nodal wrenches to body generalized forces."""

from __future__ import annotations

from FMBD.legacy_numpy_cupy.core.body import Body
from FMBD.legacy_numpy_cupy.core.forces import GeneralizedForce, NodalWrench


def project_nodal_wrench(body: Body, wrench: NodalWrench) -> GeneralizedForce:
    rec, model, xp = body.reconstruction, body.model, body.model.xp
    if rec is None:
        raise RuntimeError("body must be reconstructed before force projection")
    if wrench.force.shape != (model.n_node, 3) or wrench.moment.shape != (model.n_node, 3):
        raise ValueError("nodal force and moment must both have shape (n_node, 3)")
    translation = wrench.force.sum(axis=0)
    rotation = xp.cross(rec.xrel, wrench.force, axis=1).sum(axis=0) + wrench.moment.sum(axis=0)
    force_local, moment_local = wrench.force @ body.state.R, wrench.moment @ body.state.R
    modal = xp.einsum("nim,ni->m", model.phi_pos, force_local) + xp.einsum("nim,ni->m", model.phi_rot, moment_local)
    return GeneralizedForce(translation, rotation, modal)
