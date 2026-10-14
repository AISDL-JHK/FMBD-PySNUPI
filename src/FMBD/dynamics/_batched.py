"""Grouped rigid/modal updates preserving scalar-path noise draw ordering."""
import math
import torch
from FMBD.core.batching import grouped_bodies
from FMBD.math.so3 import exp_so3_batch


def _cap(value, maximum):
    if maximum is None:
        return value
    return value*(maximum/torch.linalg.vector_norm(value,dim=-1,keepdim=True).clamp_min(1e-30)).clamp(max=1.)


def _stack(group, values, field):
    return torch.stack([values[name][field] for name in group.names])


def _wrench(group, forces):
    R=group.state('R').transpose(-1,-2)
    ft=torch.stack([forces[name].translation for name in group.names])
    fr=torch.stack([forces[name].rotation for name in group.names])
    return torch.cat((torch.bmm(R,ft[:,:,None]).squeeze(-1),
                      torch.bmm(R,fr[:,:,None]).squeeze(-1)),dim=-1)


def _modal_force(group, forces, options):
    force=torch.stack([forces[name].modal for name in group.names])
    if options.internal_scheme!='nma':
        force=force+group.model_vectors('modal_mean_force')
    return force


def _noise(integrator, group, samples, field, sqrt_dt, rigid=False):
    xi=_stack(group,samples,field)
    if rigid:
        return group.apply('mu_rigid',group.apply('S_rigid',xi)*sqrt_dt)
    return group.apply('Bq',xi)*sqrt_dt


def step_batched(integrator, system, forces, context, force_evaluator, diagnostics):
    options=integrator.options
    midpoint=options.scheme=='midpoint_brownian'
    closure=options.internal_scheme=='meanforce_respod_ou'
    names=[name for name,body in system.bodies.items() if body.dynamic]
    samples,saved={},{}
    for name in names:
        body=system.bodies[name]
        integrator._validate_ou_interval(body)
        if options.internal_scheme!='nma' and body.model.modal_mean_force is None:
            raise ValueError(f'{options.internal_scheme} requires modal_mean_force in the BodyModel')
    # Keep scalar-path random draws in insertion order, including disabled
    # channels and variable modal/POD counts. Only linear maps are batched.
    for name in names:
        body=system.bodies[name]
        state,model=body.state,body.model
        saved[name]={field:None if getattr(state,field) is None else getattr(state,field).clone()
                     for field in ('c','R','q','a')}
        def draw(size,enabled):
            if not enabled:
                return state.q.new_zeros(size)
            return torch.randn(size,device=state.q.device,dtype=state.q.dtype,generator=integrator.generator)
        values={}
        values['pose0']=draw(6,body.dynamics.pose_brownian)
        if midpoint:
            values['pose1']=draw(6,body.dynamics.pose_brownian)
        values['modal0']=draw(model.n_mode,body.dynamics.modal_brownian)
        if midpoint:
            values['modal1']=draw(model.n_mode,body.dynamics.modal_brownian)
        if closure:
            values['ou0']=draw(model.n_respod_mode,body.dynamics.modal_brownian)
            if midpoint:
                values['ou1']=draw(model.n_respod_mode,body.dynamics.modal_brownian)
        samples[name]=values
    rigid_groups=grouped_bodies(system,dynamic_only=True,rigid=True)
    modal_groups=grouped_bodies(system,dynamic_only=True)
    half=0.5*integrator.dt if midpoint else integrator.dt
    sqrt_dt=math.sqrt(half)
    def pose_update(group,current_forces,full):
        R0,c0=_stack(group,saved,'R'),_stack(group,saved,'c')
        deta=group.apply('mu_rigid',_wrench(group,current_forces))*(integrator.dt if full else half)
        deta=deta+_noise(integrator,group,samples,'pose0',sqrt_dt,rigid=True)
        if full and midpoint:
            deta=deta+_noise(integrator,group,samples,'pose1',sqrt_dt,rigid=True)
        dc,dtheta=deta[:,:3],deta[:,3:]
        if full:
            dc=_cap(dc,options.max_translation_step)
            dtheta=_cap(dtheta,options.max_rotation_step)
        dc_global=torch.bmm(R0,dc[:,:,None]).squeeze(-1)
        group.assign(c=c0+dc_global,R=torch.bmm(R0,exp_so3_batch(dtheta)))
        return dc_global,dtheta
    def modal_update(group,current_forces,full):
        dq=group.apply('mu',_modal_force(group,current_forces,options))*(integrator.dt if full else half)
        dq=dq+_noise(integrator,group,samples,'modal0',sqrt_dt)
        if full and midpoint:
            dq=dq+_noise(integrator,group,samples,'modal1',sqrt_dt)
        if full:
            dq=_cap(dq,options.max_modal_step)
        group.assign(q=_stack(group,saved,'q')+dq)
        da=None
        if closure:
            rho=group.model_vectors('respod_ou_rho')
            sigma=group.model_vectors('respod_ou_sigma')
            if midpoint:
                rho=torch.sqrt(rho)
            innovation=sigma*torch.sqrt((1-rho.square()).clamp_min(0.))
            if full and midpoint:
                a=rho*group.state('a')+innovation*_stack(group,samples,'ou1')
            else:
                a=rho*_stack(group,saved,'a')+innovation*_stack(group,samples,'ou0')
            if full:
                da=a-_stack(group,saved,'a')
            group.assign(a=a)
        return dq,da
    if midpoint:
        # Both drifts use the accepted rigid pose, as in the scalar version.
        for group in modal_groups:
            modal_update(group,forces,False)
        for group in rigid_groups:
            pose_update(group,forces,False)
        try:
            system.reconstruct_all()
            midpoint_forces=force_evaluator(context)
            missing=[name for name in names if name not in midpoint_forces]
            if missing:
                raise KeyError(f'midpoint force evaluation omitted dynamic bodies: {missing}')
        except Exception:
            for name,values in saved.items():
                state=system.bodies[name].state
                for field,value in values.items():
                    setattr(state,field,value)
            system.reconstruct_all()
            raise
        current_forces=midpoint_forces
    else:
        current_forces=forces
    results={}
    for group in rigid_groups:
        dc,dtheta=pose_update(group,current_forces,True)
        if diagnostics:
            dc_norm=torch.linalg.vector_norm(dc,dim=-1)
            dw_norm=torch.linalg.vector_norm(dtheta,dim=-1)
            ft=torch.linalg.vector_norm(torch.stack([current_forces[n].translation for n in group.names]),dim=-1)
            fr=torch.linalg.vector_norm(torch.stack([current_forces[n].rotation for n in group.names]),dim=-1)
            for i,name in enumerate(group.names):
                results[name]={'dc_norm':dc_norm[i].detach(),'dw_norm':dw_norm[i].detach(),
                    'Ft_norm':ft[i].detach(),'Fr_norm':fr[i].detach()}
    for group in modal_groups:
        dq,da=modal_update(group,current_forces,True)
        if diagnostics:
            dq_norm=torch.linalg.vector_norm(dq,dim=-1)
            fq=torch.linalg.vector_norm(torch.stack([current_forces[n].modal for n in group.names]),dim=-1)
            da_norm=None if da is None else torch.linalg.vector_norm(da,dim=-1)
            for i,name in enumerate(group.names):
                results[name].update(dq_norm=dq_norm[i].detach(),Fq_norm=fq[i].detach())
                if da_norm is not None:
                    results[name]['da_norm']=da_norm[i].detach()
    return results
