"""Full-mass relaxed Galerkin derivatives and exact edge-bundle diagnostics.

Hard one-hot forward physics agrees with round 2 in centroid mode. Full mass
matrices are retained only to avoid dropping off-diagonal relaxation derivatives.
The edge bundle is a diagnostic constitutive model, NOT a disclosed PhySPRING ROM.
"""
from dataclasses import dataclass
import torch
from .physics import System,laplacian
from .physics_v2 import spring_force,ground_step

@dataclass
class GalerkinSystem(System):
    full_mass: torch.Tensor | None = None

    def to(self,device,dtype=None):
        values=[v.to(device=device,dtype=dtype if v.is_floating_point() else None) if v is not None else None for v in self.__dict__.values()]
        return GalerkinSystem(*values)


def mass_matrix(s):
    m=getattr(s,'full_mass',None)
    return torch.diag_embed(s.mass) if m is None else m


def project_full(s,p):
    mf=mass_matrix(s);mc=p.T@mf@p
    rest=torch.linalg.solve(mc,p.T@mf@s.rest)
    lc=p.T@laplacian(s.stiffness)@p
    weights=-lc+torch.diag_embed(lc.diagonal())
    return GalerkinSystem(rest,mc.diagonal(),weights,p.T@s.damping@p,(p.detach().T@s.fixed.to(p.dtype))>0,s.contact,mc)


def pool_full(x,mass,p):
    mc=p.T@(mass[:,None]*p)
    rhs=torch.einsum('nk,...nc->...kc',p,x*mass[:,None])
    return torch.linalg.solve(mc,rhs)


def make_bundle_force(fine,p,coarse_rest):
    """Exact P^T F(x_rest + P*u, P*v), using all ORIGINAL undirected edges.

    P may have gradients; the original edge set remains fixed. No neighbor cutoff,
    nearest-centroid direction, or deletion of within-cluster edges in backward.
    """
    i,j=torch.where(torch.triu(fine.stiffness.detach()>0,diagonal=1))
    a=p[i]-p[j]  # fine-edge incidence times P; zero for intra-cluster hard edges
    r=fine.rest[i]-fine.rest[j];l0=torch.linalg.vector_norm(r,dim=-1)
    k=fine.stiffness[i,j];d=(-fine.damping[i,j]).clamp_min(0)
    def force(x,v):
        delta=r+torch.einsum('ec,bcd->bed',a,x-coarse_rest)
        length=torch.linalg.vector_norm(delta,dim=-1).clamp_min(1e-7);normal=delta/length[...,None]
        velocity=torch.einsum('ec,bcd->bed',a,v)
        scalar=-k*(length-l0)-d*(velocity*normal).sum(-1)
        return torch.einsum('ec,bed->bcd',a,scalar[...,None]*normal)
    return force


def rollout(s,x0,v0,controls,dt,substeps,gravity=None,floor=0.,force_fn=None,checkpoint_frames=False):
    if gravity is None:gravity=x0.new_tensor([0.,0.,-9.8])
    h=dt/substeps;fixed=s.fixed[None,:,None]
    # Same hard forward diagonal inverse; full inverse retains cross-mass derivatives.
    inv=torch.linalg.inv(mass_matrix(s))
    drag=torch.exp(-h*s.damping.sum(-1).clamp_min(0)/s.mass)[:,None]
    ffun=(lambda x,v:spring_force(x,v,s)) if force_fn is None else force_fn
    def frame(x,v,c0,c1):
        for step in range(substeps):
            target=c0+(step+1)/substeps*(c1-c0)
            x=torch.where(fixed,target,x);v=torch.where(fixed,torch.zeros_like(v),v)
            vn=(v+h*(inv@ffun(x,v)+gravity))*drag
            if floor is None:xn=x+h*vn
            else:xn,vn=ground_step(x,vn,h,s.contact[0],s.contact[1],floor)
            x=torch.where(fixed,target,xn);v=torch.where(fixed,torch.zeros_like(vn),vn)
        return x,v
    x,v=x0,v0;out=[x]
    for t in range(controls.shape[1]-1):
        if checkpoint_frames and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            x,v=checkpoint(frame,x,v,controls[:,t],controls[:,t+1],use_reentrant=False)
        else:x,v=frame(x,v,controls[:,t],controls[:,t+1])
        out.append(x)
    return torch.stack(out,1)
