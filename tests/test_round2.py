import torch
from physpring.physics import System,laplacian
from physpring.physics_v2 import spring_force,ground_step,rollout,particle_impulse
from physpring.model_v2 import PhySpring
from physpring.data import make_scene
from physpring.losses_v2 import observation_loss


def system():
    x=torch.tensor([[0.,0.,1.],[1.,0.,1.]],dtype=torch.float64)
    w=torch.tensor([[0.,2.],[2.,0.]],dtype=torch.float64)
    return System(x,torch.ones(2,dtype=torch.float64),w,laplacian(w*.5),torch.zeros(2,dtype=torch.bool),torch.tensor([.5,.3,0.],dtype=torch.float64))


def test_axial_damping_does_not_damp_transverse_velocity():
    s=system();v=torch.tensor([[[0.,1.,0.],[0.,-1.,0.]]],dtype=torch.float64)
    assert torch.allclose(spring_force(s.rest[None],v,s),torch.zeros_like(v))
    v[...,0]=torch.tensor([1.,-1.]);f=spring_force(s.rest[None],v,s)
    assert (f*v).sum()<0
    assert f.sum(1).abs().max()<1e-12


def test_strain_and_hooke_units_agree():
    s=system();s.rest[1,0]=.5
    x=s.rest.clone();x[1,0]=.6
    f=spring_force(x[None],torch.zeros_like(x)[None],s)
    y=s.stiffness[0,1]*.5
    assert torch.allclose(f[0,0,0],y*(.6/.5-1))


def test_ground_time_of_impact():
    x=torch.tensor([[[0.,0.,.1]]],dtype=torch.float64);v=torch.tensor([[[1.,0.,-2.]]],dtype=torch.float64)
    xn,vn=ground_step(x,v,.1,torch.tensor(.5),torch.tensor(0.))
    assert torch.allclose(vn,torch.tensor([[[1.,0.,1.]]],dtype=torch.float64))
    assert torch.allclose(xn,torch.tensor([[[.1,0.,.05]]],dtype=torch.float64))


def test_spring_and_rollout_gradients():
    s=system();k=s.stiffness.clone().requires_grad_();s.stiffness=k
    x=s.rest.clone();x[1,0]=1.1;controls=x[None,None].expand(1,4,2,3).clone()
    def function(w):
        ss=System(s.rest,s.mass,w,s.damping,s.fixed,s.contact)
        return rollout(ss,x[None],torch.zeros_like(x)[None],controls,.01,4,gravity=torch.zeros(3,dtype=torch.float64),floor=None).square().sum()
    assert torch.autograd.gradcheck(function,(k,),eps=1e-6,atol=1e-5)


def test_checkpointed_rollout_same_gradient():
    s=system();x=s.rest[None]+torch.tensor([[[0.,0.,0.],[.1,0.,0.]]]);c=x[:,None].expand(1,3,2,3)
    values=[]
    for checkpoint in [False,True]:
        s.stiffness=s.stiffness.detach().clone().requires_grad_()
        y=rollout(s,x,torch.zeros_like(x),c,.01,8,floor=None,checkpoint_frames=checkpoint);y.square().sum().backward()
        values.append((y.detach(),s.stiffness.grad.clone()))
    assert torch.equal(values[0][0],values[1][0])
    assert torch.allclose(values[0][1],values[1][1],atol=1e-12)


def test_separate_masks_one_way_loss():
    pred=torch.tensor([[[[0.,0.,0.],[10.,0.,0.]]]],requires_grad=True)
    target=torch.tensor([[[[0.,0.,0.],[float('nan'),0.,0.]]]])
    visible=torch.tensor([[[True,False]]]);motion=torch.zeros_like(visible)
    loss,_=observation_loss(pred,target,visible,motion)
    assert loss==0  # unmatched predicted point is not penalized by one-way observed geometry.
    loss.backward();assert torch.isfinite(pred.grad).all()


def test_freeze_supervised_partition_and_singletons():
    torch.manual_seed(8);d=make_scene(size=4,frames=1,batch=1);m=PhySpring(hidden=16,merge_rule='local')
    _,_,p=m(d['initial'],False);saved=p[0].detach().clone();m.commit_last(0)
    _,_,q=m(d['initial'],True)
    assert torch.equal(saved,q[0]);assert torch.all((q[0]==0)|(q[0]==1))
    assert (q[0].sum(0)>0).all()
    fixed=d['initial'].fixed
    for row in torch.where(fixed)[0]:assert q[0][:,q[0][row].argmax()].sum()==1


def test_self_collision_conserves_momentum_pair():
    s=system();x=torch.tensor([[[0.,0.,1.],[.01,0.,1.]]],dtype=torch.float64);v=torch.tensor([[[1.,0.,0.],[-1.,0.,0.]]],dtype=torch.float64)
    result=particle_impulse(x,v,s,.02,torch.eye(2,dtype=torch.bool))
    assert result.sum(1).abs().max()<1e-10
    assert result[0,0,0]<0


def test_lifted_force_matches_full_projected_tangent():
    from physpring.reduction_audit import axial_tangent,lifted_force
    from physpring.physics import project
    d=make_scene(size=3,frames=1,batch=1);s=d['initial']
    for attr in ('rest','mass','stiffness','damping','contact'):setattr(s,attr,getattr(s,attr).double())
    p=torch.nn.functional.one_hot(torch.tensor([0,1,2,0,1,2,3,3,3]),4).double();c=project(s,p)
    lifted=torch.kron(p,torch.eye(3,dtype=torch.float64));expected=lifted.T@axial_tangent(s)@lifted
    x=c.rest[None].clone().requires_grad_()
    jac=torch.autograd.functional.jacobian(lambda xx:lifted_force(s,p,c.rest,xx,torch.zeros_like(xx)),x).reshape(12,12)
    assert torch.allclose(jac,-expected,atol=1e-10)


def test_deterministic_stage_tail_for_short_and_uneven_runs():
    from physpring.schedule import stochastic_step
    for epochs in [4,8,9,13,80]:
        for stage in range(4):
            indices=[e for e in range(epochs) if min(3,e*4//epochs)==stage]
            assert not stochastic_step(indices[-1],epochs)
    for epoch in range(80):
        stage=min(3,epoch*4//80)
        assert stochastic_step(epoch,80)==(stage>0 and (epoch-stage*20)/20<.75)


def test_transition_motion_mask_targets_next_frame():
    from physpring.masks import align_motion_mask
    motion=torch.tensor([[True,False],[False,True],[True,True]])
    finite=torch.ones_like(motion);finite[2,1]=False
    aligned=align_motion_mask(motion,finite)
    assert torch.equal(aligned,torch.tensor([[False,False],[True,False],[False,False]]))


def test_controller_position_is_updated_before_force():
    s=system();s.fixed[0]=True;s.damping.zero_()
    controls=s.rest[None,None].repeat(1,2,1,1);controls[0,1,0,0]=.1
    y=rollout(s,s.rest[None],torch.zeros_like(s.rest)[None],controls,.1,1,gravity=torch.zeros(3,dtype=torch.float64),floor=None)
    assert torch.allclose(y[0,1,1,0],torch.tensor(1.002,dtype=torch.float64))
    assert y[0,1,0,0]==.1
