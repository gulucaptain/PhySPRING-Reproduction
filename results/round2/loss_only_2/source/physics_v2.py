"""Independent axial-spring physics with reference-style damping and floor impulses.

Weights remain Hooke k [N/m]. PhysTwin strain coefficient Y maps as Y=k*l0.
No claim that scalar Galerkin projection exactly preserves nonlinear axial response.
"""
import torch
from .physics import System


def ground_step(x,v,h,restitution,friction,floor=0.):
    z=x[...,2]-floor;vz=v[...,2]
    collide=(z+h*vz<0)&(vz < -1e-4)
    e=restitution.clamp(0,1);mu=friction.clamp(0,2)
    tangent=v[...,:2];speed=torch.linalg.vector_norm(tangent,dim=-1).clamp_min(1e-6)
    factor=(1-mu*(1+e)*vz.abs()/speed).clamp_min(0)
    bounced=torch.cat([tangent*factor[...,None],(-e*vz)[...,None]],-1)
    vn=torch.where(collide[...,None],bounced,v)
    safe_vz=torch.where(collide,vz,torch.ones_like(vz))
    toi=torch.where(collide,(-z/safe_vz).clamp(0,h),torch.zeros_like(z))
    xn=x+v*toi[...,None]+vn*(h-toi[...,None])
    return xn,vn


def spring_force(x,v,system):
    delta=x[:,:,None]-x[:,None,:]
    length=torch.linalg.vector_norm(delta,dim=-1).clamp_min(1e-7)
    direction=delta/length[...,None]
    rest_length=torch.linalg.vector_norm(system.rest[:,None]-system.rest[None,:],dim=-1)
    elastic=-system.stiffness[...,None]*(length-rest_length)[...,None]*direction
    damping_edges=(-system.damping+torch.diag_embed(system.damping.diagonal())).clamp_min(0)
    relative=v[:,:,None]-v[:,None,:]
    axial_speed=(relative*direction).sum(-1)
    dissipative=-damping_edges[...,None]*axial_speed[...,None]*direction
    return (elastic+dissipative).sum(2)


def particle_impulse(x,v,system,collision_dist,excluded):
    """Symmetric pair impulses, not original degree-averaged multi-contact update.

    Distinct contribution kept optional, with explicit adjacency exclusion.
    """
    delta=x[:,:,None]-x[:,None,:];length=torch.linalg.vector_norm(delta,dim=-1).clamp_min(1e-7)
    normal=delta/length[...,None];relative=v[:,:,None]-v[:,None,:]
    speed=(relative*normal).sum(-1)
    active=(length<collision_dist)&(speed < -1e-4)&~excluded
    invmass=torch.where(system.fixed,torch.zeros_like(system.mass),1/system.mass)
    eff=(invmass[:,None]+invmass[None,:]).clamp_min(1e-8)
    e=system.contact[0].clamp(0,1);mu=system.contact[1].clamp(0,2)
    jn=-(1+e)*speed/eff
    tangential=relative-speed[...,None]*normal
    tspeed=torch.linalg.vector_norm(tangential,dim=-1).clamp_min(1e-7)
    jt=torch.minimum(tspeed/eff,mu*jn.clamp_min(0))
    impulse=jn[...,None]*normal-jt[...,None]*tangential/tspeed[...,None]
    return v+(impulse*active[...,None]).sum(2)*invmass[:,None]


def rollout(system,x0,v0,controls,dt,substeps,gravity=None,floor=0.,control_velocity='zero',self_collision=0.,checkpoint_frames=False):
    if gravity is None:gravity=x0.new_tensor([0.,0.,-9.8])
    if control_velocity not in ('zero','finite_difference'):raise ValueError(control_velocity)
    h=dt/substeps;fixed=system.fixed[None,:,None]
    drag_rate=system.damping.sum(-1).clamp_min(0)/system.mass
    drag=torch.exp(-h*drag_rate)[:,None]
    excluded=(system.stiffness.detach()>0)|torch.eye(len(system.mass),device=x0.device,dtype=torch.bool)
    def frame(x,v,c0,c1):
        cv=(c1-c0)/dt if control_velocity=='finite_difference' else torch.zeros_like(c0)
        for step in range(substeps):
            target=c0+(step+1)/substeps*(c1-c0)
            # Reference supplies the next substep's controller position before force evaluation.
            x=torch.where(fixed,target,x);v=torch.where(fixed,cv,v)
            f=spring_force(x,v,system)
            vn=(v+h*(f/system.mass[:,None]+gravity))*drag
            if self_collision:vn=particle_impulse(x,vn,system,self_collision,excluded)
            if floor is None:xn=x+h*vn
            else:xn,vn=ground_step(x,vn,h,system.contact[0],system.contact[1],floor)
            x=torch.where(fixed,target,xn);v=torch.where(fixed,cv,vn)
        return x,v
    x,v=x0,v0;out=[x]
    for t in range(controls.shape[1]-1):
        if checkpoint_frames and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            x,v=checkpoint(frame,x,v,controls[:,t],controls[:,t+1],use_reentrant=False)
        else:x,v=frame(x,v,controls[:,t],controls[:,t+1])
        out.append(x)
    return torch.stack(out,1)
