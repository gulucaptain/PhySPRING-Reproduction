from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class System:
    rest: Tensor
    mass: Tensor
    stiffness: Tensor  # symmetric positive off-diagonal edge weights
    damping: Tensor   # full damping matrix, including projected drag
    fixed: Tensor
    contact: Tensor   # [normal stiffness, normal damping, friction]

    def to(self, device, dtype=None):
        return System(*(v.to(device=device, dtype=dtype if v.is_floating_point() else None)
                        for v in self.__dict__.values()))


def laplacian(w):
    return torch.diag_embed(w.sum(-1)) - w


def project(system, p):
    """Eq. 8, hard one-hot forward; retain full M in autograd before diagonal extraction."""
    m = p.T @ (system.mass[:, None] * p)
    mass = m.diagonal()
    rest = (p.T @ (system.mass[:, None] * system.rest)) / mass[:, None].clamp_min(1e-8)
    k = p.T @ laplacian(system.stiffness) @ p
    w = -k + torch.diag_embed(k.diagonal())
    d = p.T @ system.damping @ p
    fixed = (p.detach().T @ system.fixed.to(p.dtype)) > 0
    return System(rest, mass, w, d, fixed, system.contact)


def pool_state(x, mass, p):
    coarse_mass = (p.T @ (mass[:, None] * p)).diagonal()
    return torch.einsum('nk,...nc->...kc', p, x * mass[:, None]) / coarse_mass[:, None].clamp_min(1e-8)


def lift_state(x, system, p_total, coarse_rest):
    # Constant rest offsets avoid scoring geometric quantization as dynamic motion error.
    return system.rest + torch.einsum('nk,...kc->...nc', p_total, x - coarse_rest)


def forces(x, v, system, gravity, mode='axial', floor=None):
    if mode == 'linear':
        elastic = -(laplacian(system.stiffness) @ (x - system.rest))
    elif mode == 'literal':
        elastic = -(laplacian(system.stiffness) @ x)
    elif mode == 'axial':
        delta = x[:, :, None, :] - x[:, None, :, :]
        length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-7)
        r = system.rest[:, None, :] - system.rest[None, :, :]
        rest_length = torch.linalg.vector_norm(r, dim=-1)
        elastic = -(system.stiffness * (1 - rest_length / length))[..., None] * delta
        elastic = elastic.sum(2)
    else:
        raise ValueError(mode)
    force = elastic - system.damping @ v + system.mass[:, None] * gravity
    if floor is not None:
        penetration = (floor - x[..., 2]).clamp_min(0)
        active = (x[..., 2] < floor).to(x.dtype)
        kn, cn, mu = system.contact.unbind()
        normal = (kn * penetration - cn * v[..., 2] * active).clamp_min(0)
        tangent = v[..., :2]
        friction = -mu * normal[..., None] * tangent / torch.sqrt((tangent*tangent).sum(-1, keepdim=True)+1e-6)
        force = force + torch.cat([friction, normal[..., None]], dim=-1)
    return force


def rollout(system, x0, v0, controls, dt, substeps, gravity=None, mode='axial', floor=None):
    """Semi-implicit Euler. controls: [B,T+1,N,3], interpolated within frames."""
    if gravity is None:
        gravity = x0.new_tensor([0., 0., -9.81])
    h = dt / substeps
    x, v = x0, v0
    out = [x]
    fixed = system.fixed[None, :, None]
    for t in range(controls.shape[1]-1):
        cv = (controls[:, t+1] - controls[:, t]) / dt
        for step in range(substeps):
            f = forces(x, v, system, gravity, mode, floor)
            vn = v + h * f / system.mass[:, None]
            xn = x + h * vn
            target = controls[:, t] + ((step+1)/substeps)*(controls[:, t+1]-controls[:, t])
            x, v = torch.where(fixed, target, xn), torch.where(fixed, cv, vn)
        out.append(x)
    return torch.stack(out, dim=1)


def linear_rollout_fast(system, x0, v0, controls, dt, substeps, gravity=None):
    """Exact powering of discrete semi-implicit Euler for constant per-frame controls.

    Used ONLY for no-fixed-node, no-contact linear benchmarks; not a substitute for axial dynamics.
    """
    if system.fixed.any():
        raise ValueError('fast linear path does not support fixed nodes')
    n = len(system.mass)
    eye = torch.eye(n, device=x0.device, dtype=x0.dtype)
    h = dt/substeps
    a = -laplacian(system.stiffness) / system.mass[:, None]
    b = -system.damping / system.mass[:, None]
    transition = torch.cat([torch.cat([eye+h*h*a,h*(eye+h*b)],1), torch.cat([h*a,eye+h*b],1)],0)
    if gravity is None: gravity=x0.new_tensor([0.,0.,-9.81])
    affine = torch.cat([h*h*gravity.expand(n,3),h*gravity.expand(n,3)],0)
    # Affine map augmented with a single constant coordinate per spatial dimension.
    aug = torch.zeros((2*n+1,2*n+1),device=x0.device,dtype=x0.dtype)
    # Constant gravity differs per coordinate: propagate its coefficient and multiply afterwards.
    aug[:2*n,:2*n]=transition
    aug[:n,-1]=h*h; aug[n:2*n,-1]=h; aug[-1,-1]=1
    power=torch.linalg.matrix_power(aug,substeps)
    z=torch.cat([x0-system.rest,v0],1)
    out=[x0]
    for _ in range(controls.shape[1]-1):
        z=power[:2*n,:2*n]@z+power[:2*n,-1,None]*gravity
        out.append(system.rest+z[:,:n])
    return torch.stack(out,1)
