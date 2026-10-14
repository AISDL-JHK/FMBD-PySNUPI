from dataclasses import replace
import numpy as np
import pytest
import torch
from FMBD import (BodyModel,Body,BodyState,FMBDSystem,DynamicsOptions,Simulation,
    IntegratorOptions,OverdampedFMBDIntegrator,PiecewiseProtocol,FunctionalInteraction,
    InteractionResult,NodalWrench)
from FMBD.core.projection import project_nodal_wrench, project_nodal_wrenches_batch
from FMBD.core.batching import grouped_bodies


def model(n,m,p,seed,device):
    rng=np.random.default_rng(seed)
    def tensor(value):
        return torch.as_tensor(value,device=device,dtype=torch.float64)
    phi=tensor(rng.normal(size=(6*n,m))*.03)
    X=tensor(rng.normal(size=(n,3)))
    I=tensor(np.eye(m))
    I6=tensor(np.eye(6))
    node=phi.reshape(n,6,m)
    return BodyModel(name=str(seed),X=X,Q0=tensor(np.tile(np.eye(3),(n,1,1))),
        phi=phi,phi_pos=node[:,:3],phi_rot=node[:,3:],K=I*(1+seed*.05),Gamma=I,
        mu=I*.5,S=I*.2,Bq=I*.1,Z_rigid=I6,mu_rigid=I6*.5,S_rigid=I6*.02,
        rigid_reference_center=tensor(np.zeros(3)),connectivity=np.empty((0,2),dtype=int),
        modal_mean_force=tensor(rng.normal(size=m)*.02) if p else None,
        respod_psi=tensor(rng.normal(size=(6*n,p))*.02) if p else None,
        respod_ou_rho=tensor(np.full(p,.8)) if p else None,
        respod_ou_sigma=tensor(np.full(p,.1)) if p else None,
        closure_dt_ps=.1 if p else None)


def simulation(models,batch,scheme,closure):
    system=FMBDSystem(batch_bodies=batch)
    for i,item in enumerate(models):
        state=BodyState.at_reference(item)
        state.c=item.X.new_tensor([i*2.,0.,0.])
        state.q+=.03
        if state.a is not None:
            state.a+=.02
        body=Body(item,state,dynamic=i!=3,dynamics=DynamicsOptions(
            pose_brownian=i!=2,modal_brownian=i!=1))
        system.add_body(str(i),body)
    def nodal(current,context,diagnostics):
        return InteractionResult(nodal_wrenches={name:NodalWrench(
            -.2*body.reconstruction.x,.03*torch.ones_like(body.reconstruction.x))
            for name,body in current.bodies.items()})
    system.add_interaction(FunctionalInteraction(nodal))
    gen=torch.Generator(device=models[0].X.device).manual_seed(331)
    integrator=OverdampedFMBDIntegrator(.1,IntegratorOptions(scheme=scheme,batch_bodies=batch,
        internal_scheme='meanforce_respod_ou' if closure else 'nma',
        max_translation_step=.03,max_rotation_step=.02,max_modal_step=.04),generator=gen)
    return Simulation(system,PiecewiseProtocol(key='unused',schedule=[(0.,50)],dt=.1),integrator)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('scheme',['euler_brownian','midpoint_brownian'])
@pytest.mark.parametrize('closure',[False,True])
def test_mixed_models_and_seeded_updates_match_scalar(device,scheme,closure):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('Torch CUDA unavailable')
    p=1 if closure else 0
    first=model(3,2,p,1,device)
    # Shared model, different equal-sized model, and different-sized fixed and
    # dynamic models all coexist. Rigid update groups span all modal sizes.
    models=[first,first,model(3,2,p,2,device),model(2,1,p,3,device),model(4,3,p,4,device)]
    reference=simulation(models,False,scheme,closure)
    batched=simulation(models,True,scheme,closure)
    with torch.no_grad():
        for step in range(4):
            left=reference.step(step,diagnostics=True)
            right=batched.step(step,diagnostics=True)
            for name in reference.system.bodies:
                a=reference.system.bodies[name]
                b=batched.system.bodies[name]
                for field in ('q','R','c','a'):
                    if getattr(a.state,field) is not None:
                        torch.testing.assert_close(getattr(a.state,field),getattr(b.state,field),rtol=2e-10,atol=2e-10)
                torch.testing.assert_close(a.reconstruction.x,b.reconstruction.x,rtol=2e-10,atol=2e-10)
                torch.testing.assert_close(left.forces[name].modal,right.forces[name].modal,rtol=2e-10,atol=2e-10)
                if a.dynamic:
                    for key,value in left.integrator_diagnostics[name].items():
                        torch.testing.assert_close(value,right.integrator_diagnostics[name][key],rtol=2e-10,atol=2e-10)


def test_batch_cache_updates_after_body_or_model_changes():
    a,b=model(2,1,0,1,'cpu'),model(2,1,0,2,'cpu')
    sim=simulation([a,b],True,'midpoint_brownian',False)
    group=grouped_bodies(sim.system)[0]
    before=group.apply('K',torch.ones((2,1),dtype=torch.float64))
    b.K.mul_(2)
    after=group.apply('K',torch.ones((2,1),dtype=torch.float64))
    torch.testing.assert_close(after[1],before[1]*2)
    sim.system.bodies['1'].dynamic=False
    assert grouped_bodies(sim.system,dynamic_only=True)[0].names==('0',)
    sim.system.add_body('extra',Body(a,BodyState.at_reference(a)))
    assert len(grouped_bodies(sim.system)[0].bodies)==3


def test_batched_midpoint_failure_restores_state_and_geometry():
    first=model(3,2,1,1,'cpu')
    sim=simulation([first,first],True,'midpoint_brownian',True)
    sim.initialize()
    saved={name:{key:getattr(body.state,key).clone() for key in ('q','a','R','c')}
        for name,body in sim.system.bodies.items()}
    context=sim.protocol.context(0)
    force=sim.evaluate_forces(context)[0]
    def fail(context):
        raise RuntimeError('force callback failed')
    with pytest.raises(RuntimeError,match='callback failed'):
        sim.integrator.step(sim.system,force,context,force_evaluator=fail)
    for name,body in sim.system.bodies.items():
        for key,value in saved[name].items():
            torch.testing.assert_close(getattr(body.state,key),value)
