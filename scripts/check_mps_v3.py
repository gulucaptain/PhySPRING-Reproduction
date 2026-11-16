"""CPU/Metal parity at paper-scale integration steps; two-frame diagnostic only."""
import json,sys,time
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.evaluate_v2 import restore
from physpring.physics import project,pool_state
from physpring.physics_v3 import rollout,make_bundle_force

torch.set_num_threads(1);rows=[]
for folder in [root/'results/round2_aligned/full_global_0',root/'results/round2/gpu_144']:
 model,data,cfg=restore(folder)
 with torch.no_grad():_,maps,_=model(data['initial']);p0=maps[3].detach()
 for mode in ['centroid','bundle']:
  for steps in [16,667]:
   reference=None
   for device in ['cpu','mps']:
    fine=data['initial'].to(device);p=p0.to(device);scale=torch.tensor(.95,device=device,requires_grad=True);fine.stiffness=fine.stiffness*scale
    coarse=project(fine,p);controls=pool_state(data['controls'][:,:3].to(device),fine.mass,p);x=pool_state(data['x0'].to(device),fine.mass,p);v=pool_state(data['v0'].to(device),fine.mass,p)
    f=make_bundle_force(fine,p,coarse.rest) if mode=='bundle' else None
    start=time.perf_counter();y=rollout(coarse,x,v,controls,data['dt'],steps,floor=0.,force_fn=f);loss=y[:,-1].square().mean();loss.backward()
    if device=='mps':torch.mps.synchronize()
    row=dict(case=folder.name,nodes=len(fine.mass),coarse_nodes=len(coarse.mass),batch=len(x),mode=mode,substeps=steps,device=device,seconds=time.perf_counter()-start,loss=float(loss.detach().cpu()),scale_gradient=float(scale.grad.cpu()))
    if device=='mps':row['driver_memory_bytes']=torch.mps.driver_allocated_memory()
    result=(y.detach().cpu(),scale.grad.detach().cpu())
    if reference is not None:
     row['position_max_abs_m']=float((result[0]-reference[0]).abs().max());row['scale_gradient_abs_difference']=float((result[1]-reference[1]).abs())
     assert torch.allclose(result[0],reference[0],atol=3e-5,rtol=1e-3)
     assert torch.allclose(result[1],reference[1],atol=2e-6,rtol=2e-2)
    else:reference=result
    rows.append(row);print(json.dumps(row),flush=True)
    del y,loss,f,coarse,scale,fine
    if device=='mps':torch.mps.empty_cache()
(root/'results/round3/hardware_substeps.json').write_text(json.dumps(rows,indent=2))
