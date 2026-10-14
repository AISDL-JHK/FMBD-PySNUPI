"""Projection from node-level wrenches to body generalized forces."""

from __future__ import annotations

import torch

from FMBD.core.body import Body
from FMBD.core.forces import GeneralizedForce, NodalWrench


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


def project_nodal_wrenches_batch(group, wrenches):
    for name,body in zip(group.names,group.bodies):
        wrench=wrenches[name]
        if body.reconstruction is None:
            raise RuntimeError('body must be reconstructed before force projection')
        if wrench.force.shape!=(body.model.n_node,3) or wrench.moment.shape!=(body.model.n_node,3):
            raise ValueError('nodal wrench shape does not match body model')
    force=torch.stack([wrenches[name].force for name in group.names])
    moment=torch.stack([wrenches[name].moment for name in group.names])
    xrel=torch.stack([body.reconstruction.xrel for body in group.bodies])
    translation=force.sum(dim=1)
    rotation=(torch.linalg.cross(xrel,force,dim=-1)+moment).sum(dim=1)
    R=group.state('R')
    local_force=torch.bmm(force,R)
    local_moment=torch.bmm(moment,R)
    local=torch.cat((local_force,local_moment),dim=-1).reshape(len(group.bodies),-1)
    phi=group.model_tensor('phi')
    if group.shared:
        phi=phi.expand(len(group.bodies),*phi.shape)
    modal=torch.bmm(local[:,None],phi).squeeze(1)
    return {name:GeneralizedForce(translation[i],rotation[i],modal[i]) for i,name in enumerate(group.names)}
