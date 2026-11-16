import torch
from physpring.data import make_scene
from physpring.physics import project,pool_state,lift_state
from physpring.physics_v2 import rollout as old_rollout,spring_force
from physpring.physics_v3 import project_full,pool_full,rollout,make_bundle_force
from physpring.reduction_audit import axial_tangent


def setup():
    s=make_scene(size=3,frames=2,batch=1,dtype=torch.float64)['initial'];s.contact=s.rest.new_tensor([.5,.3,0.])
    p=torch.nn.functional.one_hot(torch.tensor([0,1,2,3,3,4,4,4,4]),5).double()
    return s,p


def test_hard_projection_and_rollout_match():
    s,p=setup();a=project(s,p);b=project_full(s,p)
    for key in ['rest','mass','stiffness','damping']:assert torch.allclose(getattr(a,key),getattr(b,key),atol=1e-13)
    x=b.rest[None].clone();x[:,3,2]+=.01;c=x[:,None].expand(1,4,5,3);v=torch.zeros_like(x)
    ya=old_rollout(a,x,v,c,.01,8,floor=None);yb=rollout(b,x,v,c,.01,8,floor=None)
    assert torch.allclose(ya,yb,atol=1e-12)


def test_full_pool_translation_gradient_consistent():
    s,p=setup();p.requires_grad_();x=torch.ones(1,len(s.mass),3,dtype=p.dtype)*.37
    y=pool_full(x,s.mass,p);g=torch.autograd.grad(y.sum(),p)[0]
    # Under row-simplex perturbations, constant displacements must remain constant.
    centered=g-g.mean(-1,keepdim=True)
    assert centered.abs().max()<1e-12
    old=pool_state(x,s.mass,p);go=torch.autograd.grad(old.sum(),p)[0]
    assert (go-go.mean(-1,keepdim=True)).abs().max()>1e-3


def test_bundle_matches_dense_projected_force_and_gradients():
    s,p=setup();c=project_full(s,p);torch.manual_seed(1)
    x=(c.rest[None]+torch.randn(2,5,3,dtype=p.dtype)*.01).requires_grad_();v=torch.randn_like(x)*.02
    f=make_bundle_force(s,p,c.rest)(x,v)
    xf=s.rest+p@(x-c.rest);expected=p.T@spring_force(xf,p@v,s)
    assert torch.allclose(f,expected,atol=1e-12)
    assert torch.allclose(torch.autograd.grad(f.square().sum(),x,retain_graph=True)[0],torch.autograd.grad(expected.square().sum(),x)[0],atol=1e-10)
    assert f.sum(1).abs().max()<1e-12


def test_bundle_tangent_is_tensor_galerkin():
    s,p=setup();c=project_full(s,p);f=make_bundle_force(s,p,c.rest)
    jac=torch.autograd.functional.jacobian(lambda x:f(x,torch.zeros_like(x)),c.rest[None]).reshape(15,15)
    q=torch.kron(p,torch.eye(3,dtype=p.dtype));expected=q.T@axial_tangent(s)@q
    assert torch.allclose(jac,-expected,atol=1e-10)


def test_full_projection_soft_gradient_finite_difference():
    s,p=setup();p=(p*.96+.04/p.shape[1]).requires_grad_()
    assert torch.autograd.gradcheck(lambda q:project_full(s,q).rest,(p,),eps=1e-6,atol=1e-5)


def test_bundle_rigid_rotation_has_zero_elastic_force():
    s,p=setup();p=torch.eye(len(s.mass),dtype=p.dtype);c=project_full(s,p)
    a=torch.tensor(.4,dtype=p.dtype);r=torch.tensor([[1.,0.,0.],[0.,a.cos(),-a.sin()],[0.,a.sin(),a.cos()]],dtype=p.dtype)
    x=(s.rest@r.T)[None]
    assert make_bundle_force(s,p,c.rest)(x,torch.zeros_like(x)).abs().max()<1e-12


def test_full_rollout_gradient_is_independent_of_coordinate_origin():
    from physpring.physics import System
    s,p=setup();torch.manual_seed(9)
    controls=s.rest[None,None].repeat(1,4,1,1);controls[:,1:,0,0]+=.03
    results=[]
    for shift in [torch.zeros(3,dtype=p.dtype),torch.tensor([.2,-.3,.4],dtype=p.dtype)]:
        fine=System(s.rest+shift,s.mass,s.stiffness,s.damping,s.fixed,s.contact)
        pp=p.clone().requires_grad_();c=project_full(fine,pp)
        x=pool_full(fine.rest[None],fine.mass,pp);control=pool_full(controls+shift,fine.mass,pp)
        y=rollout(c,x,torch.zeros_like(x),control,.02,4,floor=None)
        lifted=lift_state(y,fine,pp,c.rest)
        loss=(lifted-(controls+shift)).square().mean()
        g=torch.autograd.grad(loss,pp)[0];results.append(g-g.mean(-1,keepdim=True))
    assert torch.allclose(results[0],results[1],atol=1e-11,rtol=1e-7)
