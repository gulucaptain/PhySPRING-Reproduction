import json,sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.physics import project,laplacian
from physpring.reduction_audit import axial_tangent,lifted_force
from physpring.model_v2 import PhySpring
from physpring.data import load_scene

torch.set_num_threads(1);torch.manual_seed(0)
d=load_scene(root/'data/double_lift_cloth_1_64_round2.pt');s=d['initial'].to('cpu',torch.float64)
# Exact full-N deterministic Eq.6 cannot coarsen generic distinct latent features.
h=torch.randn(68,128,dtype=torch.float64);distance=torch.cdist(h,h)
deterministic=(-distance).argmax(-1)
counts=[]
for _ in range(256):
 u=torch.rand_like(distance).clamp(1e-8,1-1e-8);labels=(-distance-torch.log(-torch.log(u))).argmax(-1)
 counts.append(int(labels.unique().numel()))
result={'literal_eq6':{'nodes':68,'deterministic_nonempty_columns':int(deterministic.unique().numel()),'random_features_note':'illustrative random features, not trained features','gumbel_nonempty_min':min(counts),'gumbel_nonempty_max':max(counts)},'tangent_audit':[]}
for strategy in ['learned','geometry','random']:
 m=PhySpring(hidden=32,strategy=strategy).double();_,_,ps=m(s,False);p=ps[0];coarse=project(s,p)
 H=axial_tangent(s);P=torch.kron(p.contiguous(),torch.eye(3,dtype=p.dtype));exact=P.T@H@P;collapsed=axial_tangent(coarse)
 relative=float((collapsed-exact).norm()/exact.norm())
 scalar=float((laplacian(coarse.stiffness)-p.T@laplacian(s.stiffness)@p).abs().max())
 z=coarse.rest[None].detach().clone().requires_grad_();v=torch.zeros_like(z)
 jac=torch.autograd.functional.jacobian(lambda zz:lifted_force(s,p,coarse.rest,zz,v),z).reshape(3*len(coarse.mass),-1)
 consistency=float((jac+exact).abs().max())
 result['tangent_audit'].append(dict(strategy=strategy,nodes=len(coarse.mass),scalar_projection_max_error=scalar,axial_tangent_relative_error=relative,lifted_force_jacobian_max_error=consistency))
(root/'results/round2/reduction_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
