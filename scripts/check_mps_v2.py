"""Round 2 CPU/Apple GPU parity; timings include device warmup."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from physpring.data import make_scene
from physpring.physics_v2 import rollout

torch.set_num_threads(1);root=Path(__file__).resolve().parents[1];rows=[]
for size,batch in [(4,2),(8,8),(12,32)]:
 d=make_scene(size=size,frames=4,batch=batch,substeps=16)
 reference=None
 for device in ['cpu','mps']:
  s=d['initial'].to(device);s.contact=torch.tensor([.5,.3,0.],device=device)
  x=d['x0'].to(device);v=d['v0'].to(device);c=d['controls'].to(device)
  s.stiffness=s.stiffness.detach().clone().requires_grad_()
  t=time.perf_counter();pred=rollout(s,x,v,c,d['dt'],8);loss=pred.square().mean();loss.backward()
  if device=='mps':torch.mps.synchronize()
  elapsed=time.perf_counter()-t
  row=dict(size=size,nodes=size*size,batch=batch,device=device,forward_backward_s=elapsed)
  result=(pred.detach().cpu(),s.stiffness.grad.cpu())
  if reference is not None:
   row['position_max_abs']=float((result[0]-reference[0]).abs().max());row['gradient_max_abs']=float((result[1]-reference[1]).abs().max())
   assert torch.allclose(result[0],reference[0],atol=2e-5,rtol=2e-4)
   assert torch.allclose(result[1],reference[1],atol=2e-6,rtol=2e-3)
  else:reference=result
  rows.append(row);print(row,flush=True)
(root/'results/round2/hardware_benchmark.json').write_text(json.dumps(rows,indent=2))
