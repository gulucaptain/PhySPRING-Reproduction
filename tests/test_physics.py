import torch
from physpring.physics import System,laplacian,project,rollout,linear_rollout_fast,forces
from physpring.data import make_scene
from physpring.model import PhySpring


def test_projection_structure_and_energy():
    s=make_scene(size=3,frames=1,batch=1,dtype=torch.float64)['initial']
    p=torch.nn.functional.one_hot(torch.tensor([0,1,2,0,1,2,3,3,3]),4).double()
    c=project(s,p)
    assert torch.allclose(c.mass.sum(),s.mass.sum())
    assert torch.all(c.mass>0)
    assert torch.allclose(laplacian(c.stiffness),p.T@laplacian(s.stiffness)@p)
    y=torch.randn(4,3,dtype=torch.float64)
    assert torch.allclose(((p@y)*(laplacian(s.stiffness)@(p@y))).sum(),(y*(laplacian(c.stiffness)@y)).sum())
    assert torch.linalg.eigvalsh(c.damping).min()>=-1e-10


def test_axial_rest_equilibrium_and_net_force():
    s=make_scene(size=3,frames=1,batch=1,dtype=torch.float64)['initial']
    zero=torch.zeros_like(s.rest)[None]
    f=forces(s.rest[None],zero,s,torch.zeros(3,dtype=torch.float64))
    assert f.abs().max()<1e-10
    f=forces(s.rest[None]+torch.randn_like(zero)*.02,zero,s,torch.zeros(3,dtype=torch.float64))
    assert f.sum(1).abs().max()<1e-10


def test_gradient_against_finite_difference():
    d=make_scene(size=3,frames=2,batch=1,dtype=torch.float64)
    s=d['initial']; scale=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
    def fun(a):
        ss=System(s.rest,s.mass,s.stiffness*a,s.damping,s.fixed,s.contact)
        out=rollout(ss,d['x0'],d['v0'],d['controls'],d['dt'],4)
        return out[:,-1].square().sum()
    grad=torch.autograd.grad(fun(scale),scale)[0]
    eps=1e-5; numerical=(fun(scale.detach()+eps)-fun(scale.detach()-eps))/(2*eps)
    assert torch.allclose(grad,numerical,atol=1e-7,rtol=1e-4)


def test_powered_euler_matches_substeps():
    d=make_scene(size=3,frames=3,batch=2,dtype=torch.float64);s=d['initial'];s.fixed=torch.zeros_like(s.fixed)
    a=rollout(s,d['x0'],d['v0'],d['controls'],d['dt'],32,mode='linear')
    b=linear_rollout_fast(s,d['x0'],d['v0'],d['controls'],d['dt'],32)
    assert torch.allclose(a,b,atol=1e-10,rtol=1e-10)


def test_hierarchy_and_ste_gradient():
    torch.manual_seed(2);d=make_scene(size=4,frames=2,batch=1)
    model=PhySpring(hidden=16)
    systems,maps,ps=model(d['initial'],stochastic=True)
    assert [len(s.mass) for s in systems]==[16,10,6,5]
    for p,s in zip(ps,systems[1:]):
        assert torch.allclose(p.sum(1),torch.ones(len(p)))
        assert (p.sum(0)>0).all()
        assert torch.allclose(s.mass.sum(),torch.tensor(1.))
        assert s.fixed.sum()==2
    loss=sum((s.rest**2).sum() for s in systems[1:]);loss.backward()
    assert model.node_enc[0].weight.grad.abs().sum()>0
    model.commit(d['initial'],0)
    a=model(d['initial'],True)[2][0]; b=model(d['initial'],True)[2][0]
    assert torch.equal(a,b)


def test_large_finite_gradients_are_not_zeroed():
    from physpring.numerics import safe_clip_grad_norm
    p=torch.nn.Parameter(torch.ones(3));p.grad=torch.tensor([1e26,-2e26,3e26])
    norm=safe_clip_grad_norm([p],10.)
    assert torch.isfinite(norm)
    assert torch.allclose(p.grad.norm(),torch.tensor(10.),atol=1e-5)
    assert p.grad[0]>0 and p.grad[1]<0
