"""Projection from node-level wrenches to body generalized forces."""

from __future__ import annotations

import torch

from FMBD.legacy_torch.core.body import Body
from FMBD.legacy_torch.core.forces import GeneralizedForce, NodalWrench


def project_nodal_wrench(body: Body, wrench: NodalWrench) -> GeneralizedForce:
    """Project a global nodal wrench to translation, rotation and modal force.

    ``body.reconstruction`` must be current.  This is deliberately a core
    mechanics operation rather than an interaction responsibility.
    """
    rec = body.reconstruction
    if rec is None:
        raise RuntimeError("body must be reconstructed before force projection")
    if wrench.force.shape != (body.model.n_node, 3) or wrench.moment.shape != (body.model.n_node, 3):
        raise ValueError("nodal force and moment must both have shape (n_node, 3)")
    translation = wrench.force.sum(dim=0)
    rotation = torch.linalg.cross(rec.xrel, wrench.force, dim=1).sum(dim=0) + wrench.moment.sum(dim=0)
    force_local = wrench.force @ body.state.R
    moment_local = wrench.moment @ body.state.R
    modal = torch.einsum("nim,ni->m", body.model.phi_pos, force_local)
    modal = modal + torch.einsum("nim,ni->m", body.model.phi_rot, moment_local)
    return GeneralizedForce(translation=translation, rotation=rotation, modal=modal)
