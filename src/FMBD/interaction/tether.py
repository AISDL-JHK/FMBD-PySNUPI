"""Freely rotating ssDNA tethers with a finite-extension FJC approximation.

No endpoint orientation is read and no nodal couple is applied. Attachment
forces still generate body torque through their moment arms. The inverse
Langevin approximation is y=x*(3-x*x)/(1-x*x), not a fitted ssDNA law.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from FMBD.core.forces import NodalWrench
from .base import InteractionResult
from .single_strand import SingleStrandDNAParameters


@dataclass(frozen=True)
class FJCTetherParameters:
    unpaired_nt: int
    contour_length: float
    kuhn_length: float = 1.5
    temperature: float = 300.0
    regularization_fraction: float = 0.98

    def __post_init__(self):
        if not isinstance(self.unpaired_nt,int) or self.unpaired_nt < 1:
            raise ValueError('unpaired_nt must be a positive integer')
        if not all(math.isfinite(v) and v>0 for v in (self.contour_length,self.kuhn_length,self.temperature)):
            raise ValueError('Lengths and temperature must be finite and positive')
        if not 0 < self.regularization_fraction < 1:
            raise ValueError('regularization_fraction must lie in (0,1)')

    @classmethod
    def from_nucleotides(cls, unpaired_nt=16, *, kuhn_length=1.5, temperature=300.):
        contour=SingleStrandDNAParameters.from_nucleotides(unpaired_nt).contour_length
        return cls(unpaired_nt,contour,kuhn_length,temperature)


@dataclass
class SingleStrandTetherInteraction:
    body_a: str
    body_b: str
    pairs: object
    parameters: FJCTetherParameters
    name: str = 'single_strand_tethers'

    def initialize(self, system, context):
        self._pairs=np.asarray(self.pairs,dtype=np.int64)
        if self._pairs.ndim!=2 or self._pairs.shape[1]!=2 or not len(self._pairs):
            raise ValueError('pairs must have shape (n_linker,2)')
        a,b=system.bodies[self.body_a],system.bodies[self.body_b]
        if a.model.X.device!=b.model.X.device or a.model.X.dtype!=b.model.X.dtype:
            raise ValueError('Tether endpoints must share device and dtype')
        if (self._pairs<0).any() or (self._pairs[:,0]>=a.model.n_node).any() or (self._pairs[:,1]>=b.model.n_node).any():
            raise IndexError('Tether node out of bounds')
        self._node_a=torch.as_tensor(self._pairs[:,0],device=a.model.X.device)
        self._node_b=torch.as_tensor(self._pairs[:,1],device=b.model.X.device)

    def compute(self, system, context, diagnostics=False):
        return self.compute_batch([self],system,context,diagnostics)[0]

    @staticmethod
    def compute_batch(interactions, system, context, diagnostics=False):
        p=interactions[0].parameters
        chords=[]
        for item in interactions:
            if item.parameters!=p:
                raise ValueError('Tether batch materials must match')
            a,b=system.bodies[item.body_a],system.bodies[item.body_b]
            chords.append(b.reconstruction.x[item._node_b]-a.reconstruction.x[item._node_a])
        d=torch.cat(chords)
        lengths=torch.linalg.vector_norm(d,dim=-1)
        x=lengths/p.contour_length
        z=x.clamp(max=p.regularization_fraction)
        z2=z.square()
        denominator=1-z2
        scale=0.013806504*p.temperature/p.kuhn_length  # kBT/b, pN
        y=z*(3-z2)/denominator
        derivative=(3+z2.square())/denominator.square()
        # Tangent continuation near contour length prevents singular numerical
        # operations; diagnostics expose excursions rather than hiding them.
        tension=scale*(y+derivative*(x-z))
        force=d*(tension/lengths.clamp_min(1e-30))[:,None]
        energy=None
        if diagnostics:
            energy=scale*p.contour_length*(0.5*z2-torch.log1p(-z2)+y*(x-z)+0.5*derivative*(x-z).square())
        results=[]
        offset=0
        for item in interactions:
            a,b=system.bodies[item.body_a],system.bodies[item.body_b]
            wa=NodalWrench.zeros(a.model.n_node,device=d.device,dtype=d.dtype)
            wb=NodalWrench.zeros(b.model.n_node,device=d.device,dtype=d.dtype)
            stop=offset+len(item._pairs)
            chunk=slice(offset,stop)
            wa.force.index_add_(0,item._node_a,force[chunk])
            wb.force.index_add_(0,item._node_b,-force[chunk])
            nodal={item.body_a:wa}
            if item.body_b in nodal:
                nodal[item.body_b].add_(wb)
            else:
                nodal[item.body_b]=wb
            results.append(InteractionResult(nodal_wrenches=nodal,
                energy=energy[chunk].sum() if diagnostics else None,
                diagnostics={'lengths_nm':lengths[chunk],'extension_fraction':x[chunk],
                    'regularized':x[chunk]>p.regularization_fraction,'material':asdict(p)} if diagnostics else {}))
            offset=stop
        return results
