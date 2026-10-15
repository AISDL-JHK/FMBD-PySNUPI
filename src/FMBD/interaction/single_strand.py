"""PySNUPI-compatible ssDNA material and corotational 7-DOF beam energy.

The nucleotide count refers to unpaired linker bases, excluding attachment
bases. Angular reference frames are captured at initialization; axial rest
length is determined by the material law, never by initial placement.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from FMBD.core.forces import NodalWrench
from FMBD.math.so3 import skew, exp_so3, exp_so3_batch
from .base import InteractionResult


@dataclass(frozen=True)
class SingleStrandDNAParameters:
    unpaired_nt: int
    contour_length: float
    persistence_length: float
    reference_length: float
    EA_low: float
    EA_high: float
    slope: float
    EI: float
    GJ: float

    @classmethod
    def from_nucleotides(cls, unpaired_nt=16):
        if not isinstance(unpaired_nt, int) or unpaired_nt < 1:
            raise ValueError('unpaired_nt must be a positive integer')
        count = unpaired_nt + 2
        decay = math.exp(-0.20 * (count - 3))
        contour = (0.38 * decay + 0.68 * (1 - decay)) * (count - 1)
        persistence = 5.4 * (contour - 0.38) * math.exp(-0.21 * (contour - 0.38)) + 0.67
        eta = 2.5 - 0.75 * contour / 0.67
        reference = contour * math.sqrt((eta + math.sqrt(eta**2 + 14)) / 7)
        return cls(unpaired_nt, contour, persistence, reference, 15.0, 710.0,
            80 * math.log(0.072 * persistence + 1.16),
            1.3806504e-23 * 1e21 * 300 * persistence,
            2 + 13 * math.exp(-0.3 * (contour - 2 * 0.38)))


def _log(R):
    # Stable derivative at identity, where acos(trace(R)) is singular.
    v = 0.5 * torch.stack((R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]))
    s2 = torch.dot(v, v)
    s = s2.clamp_min(1e-30).sqrt()
    c = (torch.trace(R) - 1) * 0.5
    factor = torch.where(s2 < 1e-12, 1 + s2 / 6, torch.atan2(s, c) / s)
    return factor * v


def _skew_batch(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack((z, -v[...,2], v[...,1], v[...,2], z, -v[...,0],
                        -v[...,1], v[...,0], z), dim=-1).reshape(*v.shape[:-1],3,3)


def _log_batch(R):
    v = 0.5 * torch.stack((R[:,2,1]-R[:,1,2],R[:,0,2]-R[:,2,0],R[:,1,0]-R[:,0,1]),dim=-1)
    s2 = (v*v).sum(dim=-1)
    s = s2.clamp_min(1e-30).sqrt()
    c = (R.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*0.5
    return v * torch.where(s2<1e-12,1+s2/6,torch.atan2(s,c)/s)[:,None]


def _left_jacobian(phi, inverse=False):
    t2=(phi*phi).sum(-1)
    t=t2.clamp_min(1e-30).sqrt()
    K=_skew_batch(phi)
    I=torch.eye(3,device=phi.device,dtype=phi.dtype)[None]
    if inverse:
        coefficient=torch.where(t2<1e-8,1/12+t2/720+t2*t2/30240,
            (1-0.5*t/torch.tan(0.5*t).clamp_min(1e-30))/t2.clamp_min(1e-30))
        return I-0.5*K+coefficient[:,None,None]*(K@K)
    a=torch.where(t2<1e-8,0.5-t2/24+t2*t2/720,
        (1-torch.cos(t))/t2.clamp_min(1e-30))
    b=torch.where(t2<1e-8,1/6-t2/120+t2*t2/5040,
        (t-torch.sin(t))/(t2*t).clamp_min(1e-30))
    return I+a[:,None,None]*K+b[:,None,None]*(K@K)


def _log_differential(R, variation, phi):
    spin=variation@R[:,None].transpose(-1,-2)
    omega=0.5*torch.stack((spin[:,:,2,1]-spin[:,:,1,2],
        spin[:,:,0,2]-spin[:,:,2,0],spin[:,:,1,0]-spin[:,:,0,1]),dim=-1)
    return (_left_jacobian(phi,inverse=True)[:,None]@omega[...,None]).squeeze(-1)


@dataclass
class SingleStrandBeamInteraction:
    body_a: str
    body_b: str
    pairs: object
    parameters: SingleStrandDNAParameters
    name: str = 'single_strand_beams'

    def initialize(self, system, context):
        self._pairs = np.asarray(self.pairs, dtype=np.int64)
        if self._pairs.ndim != 2 or self._pairs.shape[1] != 2 or not len(self._pairs):
            raise ValueError('pairs must have shape (n_linker, 2)')
        a, b = system.bodies[self.body_a], system.bodies[self.body_b]
        self._frames = []
        for i, j in self._pairs:
            if not (0 <= i < a.model.n_node and 0 <= j < b.model.n_node):
                raise IndexError('linker node out of bounds')
            d = b.reconstruction.x[j] - a.reconstruction.x[i]
            x = d / torch.linalg.norm(d)
            auxiliary = a.reconstruction.Qglobal[i][:, 1]
            y = auxiliary - torch.dot(auxiliary, x) * x
            if torch.linalg.norm(y).item() < 1e-8:
                auxiliary = a.reconstruction.Qglobal[i][:, 2]
                y = auxiliary - torch.dot(auxiliary, x) * x
            y = y / torch.linalg.norm(y)
            frame = torch.stack((x, y, torch.linalg.cross(x, y)), dim=1)
            self._frames.append((a.reconstruction.Qglobal[i].T @ frame,
                                 b.reconstruction.Qglobal[j].T @ frame))
        nodes, weights = np.polynomial.legendre.leggauss(32)
        self._nodes = a.model.X.new_tensor(nodes)
        self._weights = a.model.X.new_tensor(weights)
        self._node_a=torch.as_tensor(self._pairs[:,0],device=a.model.X.device)
        self._node_b=torch.as_tensor(self._pairs[:,1],device=a.model.X.device)
        self._frame_a=torch.stack([v[0] for v in self._frames])
        self._frame_b=torch.stack([v[1] for v in self._frames])
        self._basis=torch.eye(3,device=a.model.X.device,dtype=a.model.X.dtype)
        self._spin_basis=_skew_batch(self._basis)

    def _axial_energy(self, length):
        p = self.parameters
        ref = length.new_tensor(p.reference_length)
        lo, hi = torch.minimum(length, ref), torch.maximum(length, ref)
        width = p.contour_length / abs(p.slope)
        boundaries = torch.cat((lo[None], hi[None], length.new_tensor(
            [p.contour_length + v * width for v in (-40,-8,0,8,40)]),
            lo + length.new_tensor([8,40]) * width,
            hi - length.new_tensor([8,40]) * width)).clamp(min=lo, max=hi).sort().values
        half = (boundaries[1:] - boundaries[:-1]) * 0.5
        sample = (boundaries[1:] + boundaries[:-1])[:,None] * 0.5 + half[:,None] * self._nodes
        EA = p.EA_low + (p.EA_high-p.EA_low) * torch.sigmoid(p.slope * (sample/p.contour_length-1))
        integral = (half * (self._weights * EA * (sample-ref)/ref).sum(dim=1)).sum()
        return torch.where(length >= ref, integral, -integral)

    def _energy(self, xa, xb, Qa, Qb, frames):
        p = self.parameters
        d = xb - xa
        length = torch.linalg.norm(d)
        Ha, Hb = Qa @ frames[0], Qb @ frames[1]
        midpoint = Ha @ exp_so3(0.5 * _log(Ha.T @ Hb))
        axis = d / length
        old = midpoint[:,0]
        cross = torch.linalg.cross(old, axis)
        K = skew(cross)
        # Minimal rotation maps the midpoint axial axis onto the chord.
        frame = (torch.eye(3, device=d.device, dtype=d.dtype) + K + K @ K / (1+torch.dot(old,axis)).clamp_min(1e-10)) @ midpoint
        ra, rb = _log(frame.T @ Ha), _log(frame.T @ Hb)
        angular = p.GJ / (2*p.reference_length) * (ra[0]-rb[0])**2
        angular += 2*p.EI/p.reference_length * (ra[1:]**2 + rb[1:]**2 + ra[1:]*rb[1:]).sum()
        return self._axial_energy(length) + angular

    def compute(self, system, context, diagnostics=False):
        """Analytic batch nodal response, with no potential quadrature in drift.

        Nine virtual variations (chord xyz and two endpoint left rotations)
        propagate through the corotational frame via explicit SO(3)
        differentials. This preserves the original angular potential while
        avoiding reverse-mode autograd and one graph per linker.
        """
        return self.compute_batch([self], system, context, diagnostics)[0]

    @staticmethod
    def compute_batch(interactions, system, context, diagnostics=False):
        """Evaluate compatible connectors together, retaining individual results.

        Endpoint gathering and scattering use each body's own node indices;
        body shapes and node counts need not match. Material, device and dtype
        must match across this batch.
        """
        first = interactions[0]
        p = first.parameters
        chords, frames_a, frames_b = [], [], []
        for interaction in interactions:
            a, b = system.bodies[interaction.body_a], system.bodies[interaction.body_b]
            if interaction.parameters != p:
                raise ValueError('Linker batch materials must match')
            if a.model.X.device != first._basis.device or b.model.X.device != first._basis.device:
                raise ValueError('Linker batch devices must match')
            if a.model.X.dtype != first._basis.dtype or b.model.X.dtype != first._basis.dtype:
                raise ValueError('Linker batch dtypes must match')
            chords.append(b.reconstruction.x[interaction._node_b]-a.reconstruction.x[interaction._node_a])
            frames_a.append(a.reconstruction.Qglobal[interaction._node_a]@interaction._frame_a)
            frames_b.append(b.reconstruction.Qglobal[interaction._node_b]@interaction._frame_b)
        d = torch.cat(chords)
        Ha, Hb = torch.cat(frames_a), torch.cat(frames_b)
        lengths = torch.linalg.norm(d,dim=-1)
        n = d/lengths[:,None]
        self = first
        count=d.shape[0]
        # Variations are derivatives with respect to [d, alpha_A, alpha_B].
        dd=d.new_zeros((count,9,3))
        dd[:,:3]=self._basis
        dHa=d.new_zeros((count,9,3,3))
        dHb=torch.zeros_like(dHa)
        dHa[:,3:6]=self._spin_basis[None]@Ha[:,None]
        dHb[:,6:9]=self._spin_basis[None]@Hb[:,None]
        relative=Ha.transpose(-1,-2)@Hb
        drelative=dHa.transpose(-1,-2)@Hb[:,None]+Ha[:,None].transpose(-1,-2)@dHb
        phi=_log_batch(relative)
        dphi=_log_differential(relative,drelative,phi)
        half=0.5*phi
        E=exp_so3_batch(half)
        omega=(_left_jacobian(half)[:,None]@(0.5*dphi)[...,None]).squeeze(-1)
        dE=_skew_batch(omega)@E[:,None]
        midpoint=Ha@E
        dmidpoint=dHa@E[:,None]+Ha[:,None]@dE
        old=midpoint[:,:,0]
        dold=dmidpoint[:,:,:,0]
        dn=(dd-n[:,None]*(dd*n[:,None]).sum(-1,keepdim=True))/lengths[:,None,None]
        cross=torch.linalg.cross(old,n)
        dcross=torch.linalg.cross(dold,n[:,None],dim=-1)+torch.linalg.cross(old[:,None],dn,dim=-1)
        cosine=(old*n).sum(-1)
        dcosine=(dold*n[:,None]).sum(-1)+(old[:,None]*dn).sum(-1)
        K=_skew_batch(cross)
        dK=_skew_batch(dcross)
        denominator=(1+cosine).clamp_min(1e-10)
        P=self._basis[None]+K+(K@K)/denominator[:,None,None]
        dP=dK+(dK@K[:,None]+K[:,None]@dK)/denominator[:,None,None,None]
        dP-= (K@K)[:,None]*dcosine[:,:,None,None]/denominator[:,None,None,None]**2
        frame=P@midpoint
        dframe=dP@midpoint[:,None]+P[:,None]@dmidpoint
        Za=frame.transpose(-1,-2)@Ha
        Zb=frame.transpose(-1,-2)@Hb
        dZa=dframe.transpose(-1,-2)@Ha[:,None]+frame[:,None].transpose(-1,-2)@dHa
        dZb=dframe.transpose(-1,-2)@Hb[:,None]+frame[:,None].transpose(-1,-2)@dHb
        ra,rb=_log_batch(Za),_log_batch(Zb)
        dra=_log_differential(Za,dZa,ra)
        drb=_log_differential(Zb,dZb,rb)
        ma=2*p.EI/p.reference_length*(2*ra+rb)
        mb=2*p.EI/p.reference_length*(2*rb+ra)
        ma[:,0]=p.GJ/p.reference_length*(ra[:,0]-rb[:,0])
        mb[:,0]=-ma[:,0]
        gradient=(dra*ma[:,None]+drb*mb[:,None]).sum(-1)
        EA=p.EA_low+(p.EA_high-p.EA_low)*torch.sigmoid(p.slope*(lengths/p.contour_length-1))
        tension=EA*(lengths-p.reference_length)/p.reference_length
        force=gradient[:,:3]+tension[:,None]*n
        angular = None
        if diagnostics:
            angular=p.GJ/(2*p.reference_length)*(ra[:,0]-rb[:,0])**2
            angular+=2*p.EI/p.reference_length*(ra[:,1:]**2+rb[:,1:]**2+ra[:,1:]*rb[:,1:]).sum(-1)
        results = []
        offset = 0
        for interaction in interactions:
            a, b = system.bodies[interaction.body_a], system.bodies[interaction.body_b]
            wa = NodalWrench.zeros(a.model.n_node, device=a.model.X.device, dtype=a.model.X.dtype)
            wb = NodalWrench.zeros(b.model.n_node, device=b.model.X.device, dtype=b.model.X.dtype)
            stop = offset + len(interaction._pairs)
            chunk = slice(offset, stop)
            wa.force.index_add_(0,interaction._node_a,force[chunk])
            wb.force.index_add_(0,interaction._node_b,-force[chunk])
            wa.moment.index_add_(0,interaction._node_a,-gradient[chunk,3:6])
            wb.moment.index_add_(0,interaction._node_b,-gradient[chunk,6:9])
            energy = None
            if diagnostics:
                energy=angular[chunk].sum()+torch.stack([interaction._axial_energy(length) for length in lengths[chunk]]).sum()
            results.append(InteractionResult(nodal_wrenches={interaction.body_a:wa, interaction.body_b:wb},
                energy=energy, diagnostics={'lengths_nm':lengths[chunk], 'material':asdict(p)} if diagnostics else {}))
            offset = stop
        return results
