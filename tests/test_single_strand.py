from types import SimpleNamespace
import numpy as np
import torch
from FMBD.interaction import SingleStrandBeamInteraction, SingleStrandDNAParameters
from FMBD.math.so3 import exp_so3, exp_so3_batch


def test_analytic_linker_wrenches_preserve_energy_and_balance():
    torch.manual_seed(31)
    p=SingleStrandDNAParameters.from_nucleotides(16)
    np.testing.assert_allclose([p.contour_length,p.reference_length],[11.306085951323894,3.4908421172261326])
    X=torch.tensor([[0.,0.,0.],[0.,2.,0.],[0.,0.,2.],[0.,2.,2.]],dtype=torch.float64)
    identity=torch.eye(3,dtype=torch.float64).expand(4,3,3).clone()
    a=SimpleNamespace(model=SimpleNamespace(X=X,n_node=4),reconstruction=SimpleNamespace(x=X.clone(),Qglobal=identity.clone()))
    b=SimpleNamespace(model=a.model,reconstruction=SimpleNamespace(x=X+X.new_tensor([4.,0.,0.]),Qglobal=identity.clone()))
    system=SimpleNamespace(bodies={'a':a,'b':b})
    linker=SingleStrandBeamInteraction('a','b',[[i,i] for i in range(4)],p)
    linker.initialize(system,None)
    for trial in range(3):
        if trial:
            for body in (a,b):
                body.reconstruction.x+=0.1*torch.randn_like(X)
                body.reconstruction.Qglobal=exp_so3_batch(0.2*torch.randn_like(X))@body.reconstruction.Qglobal
        with torch.no_grad():
            response=linker.compute(system,None)
        for i,frames in enumerate(linker._frames):
            xa=a.reconstruction.x[i].clone().requires_grad_(True)
            xb=b.reconstruction.x[i].clone().requires_grad_(True)
            da=X.new_zeros(3,requires_grad=True)
            db=X.new_zeros(3,requires_grad=True)
            energy=linker._energy(xa,xb,exp_so3(da)@a.reconstruction.Qglobal[i],
                exp_so3(db)@b.reconstruction.Qglobal[i],frames)
            expected=torch.autograd.grad(energy,(xa,xb,da,db))
            wa,wb=response.nodal_wrenches['a'],response.nodal_wrenches['b']
            for actual,gradient in zip((wa.force[i],wb.force[i],wa.moment[i],wb.moment[i]),expected):
                torch.testing.assert_close(actual,-gradient,atol=1e-8,rtol=1e-8)
        total_force=wa.force.sum(0)+wb.force.sum(0)
        total_torque=(torch.linalg.cross(a.reconstruction.x,wa.force).sum(0)+wa.moment.sum(0)
            +torch.linalg.cross(b.reconstruction.x,wb.force).sum(0)+wb.moment.sum(0))
        torch.testing.assert_close(total_force,torch.zeros(3,dtype=X.dtype),atol=1e-9,rtol=0)
        torch.testing.assert_close(total_torque,torch.zeros(3,dtype=X.dtype),atol=1e-8,rtol=0)
