"""Translation invariance and full/diagonal soft-extension diagnostics."""
import json,sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.evaluate_v2 import restore
from physpring.physics import pool_state,laplacian
from physpring.physics_v3 import pool_full
rows=[];torch.set_num_threads(1)
for name in ['full_global','full_local','full_geometry','full_random']:
 for seed in range(3):
  model,d,cfg=restore(root/'results/round2_aligned'/f'{name}_{seed}');base=d['initial']
  with torch.no_grad():_,maps,_=model(base)
  p=maps[3].detach().requires_grad_();const=torch.ones(1,len(base.mass),3)*.37
  r=dict(strategy=name,seed=seed)
  for method,fun in [('diagonal',pool_state),('full',pool_full)]:
   y=fun(const,base.mass,p);g=torch.autograd.grad(y.sum(),p)[0];tangent=g-g.mean(-1,keepdim=True)
   r[method+'_row_simplex_gradient_max']=float(tangent.abs().max())
   r[method+'_constant_forward_error']=float((y-.37).detach().abs().max())
  rows.append(r)
(root/'results/round3/projection_gradient_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows[:3],indent=2))
