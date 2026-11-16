"""Audits scalar Galerkin matrices versus the actual 3D axial tangent."""
import torch
from .physics_v2 import spring_force


def axial_tangent(system):
    n=len(system.mass);h=system.rest.new_zeros((n,3,n,3))
    for i in range(n):
        for j in range(i+1,n):
            delta=system.rest[i]-system.rest[j]
            length=torch.linalg.vector_norm(delta)
            if float(length.detach())<1e-10:continue
            normal=delta/length;block=system.stiffness[i,j]*normal[:,None]*normal[None,:]
            h[i,:,i,:]=h[i,:,i,:]+block;h[j,:,j,:]=h[j,:,j,:]+block
            h[i,:,j,:]=h[i,:,j,:]-block;h[j,:,i,:]=h[j,:,i,:]-block
    return h.reshape(3*n,3*n)


def lifted_force(fine,p,coarse_rest,x,v):
    # Diagnostic only: this still evaluates all fine interactions, so it is not a fast ROM.
    xf=fine.rest+p@(x-coarse_rest);vf=p@v
    return p.T@spring_force(xf,vf,fine)
