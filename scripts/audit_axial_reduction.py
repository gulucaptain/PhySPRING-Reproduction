"""Same fine system, same frozen map: isolate nonlinear constitutive reduction.

No fitting and no future-data selection. Oracle uses future observations only as
an explicitly non-predictive representation diagnostic. All modes retain the same
coarse contact/drag, so exact bundle claims apply to internal spring forces only.
"""
import json,sys,time,argparse
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.evaluate_v2 import restore
from physpring.physics import System,project,pool_state,lift_state
from physpring.physics_v2 import spring_force
from physpring.physics_v3 import rollout,make_bundle_force
from physpring.reduction_audit import axial_tangent
from physpring.experiment_v2 import metrics

parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['round2_aligned','round3'],default='round2_aligned');parser.add_argument('--levels',type=int,nargs='+',default=[1,2,3]);args=parser.parse_args()
if args.group=='round3':
 from physpring.evaluate_v3 import restore
torch.set_num_threads(1);rows=[];started=time.perf_counter()
common=None;reference=None
for name in ['full_global','full_local','full_geometry','full_random']:
 for seed in range(3):
  folder=root/'results'/args.group/f'{name}_{seed}';model,d,cfg=restore(folder);fine=d['initial'];count=d['object_count'];tf=d['train_frames'];mask=d['observation_mask'];frames=cfg['actual_frames']
  with torch.no_grad():
   _,maps,_=model(fine)
   if reference is None:
    reference=rollout(fine,d['x0'],d['v0'],d['controls'],d['dt'],32,floor=0.)
    hf=axial_tangent(fine).double()
   for level in args.levels:
    p=maps[level];coarse=project(fine,p);n=len(coarse.mass);q=torch.kron(p.double(),torch.eye(3,dtype=torch.double));exact=q.T@hf@q;hc=axial_tangent(coarse).double();free=(~coarse.fixed).repeat_interleave(3)
    tangent_error=float((hc[free][:,free]-exact[free][:,free]).norm()/exact[free][:,free].norm())
    # Best nonnegative scalar for each OFF-DIAGONAL fixed centroid direction.
    opt=coarse.stiffness.clone();blocks=exact.reshape(n,3,n,3);irreducible=0.;total=0.
    for i in range(n):
     for j in range(i+1,n):
      block=-blocks[i,:,j,:];delta=coarse.rest[i]-coarse.rest[j];normal=delta.double()/delta.double().norm().clamp_min(1e-10)
      value=(normal@block@normal).clamp_min(0);approx=value*normal[:,None]*normal[None,:]
      opt[i,j]=opt[j,i]=value;irreducible+=float((block-approx).square().sum());total+=float(block.square().sum())
    matched=System(coarse.rest,coarse.mass,opt,coarse.damping,coarse.fixed,coarse.contact)
    # Full-fine trajectory projected into this fixed reduced subspace.
    state=pool_state(reference,fine.mass,p);vel=torch.zeros_like(state);vel[:,1:]=(state[:,1:]-state[:,:-1])/d['dt'];vel[:,:,coarse.fixed]=0
    xb=state.reshape(-1,n,3);vb=vel.reshape(-1,n,3);bundle=make_bundle_force(fine,p,coarse.rest)
    zero=torch.zeros_like(vb);exact_f=bundle(xb,zero);centroid_f=spring_force(xb,zero,coarse);exact_d=bundle(xb,vb)-exact_f;centroid_d=spring_force(xb,vb,coarse)-centroid_f
    def relative(a,b):return float((a[:,~coarse.fixed]-b[:,~coarse.fixed]).norm()/b[:,~coarse.fixed].norm().clamp_min(1e-12))
    # Least-squares displacement oracle under the common evaluation mask.
    w=mask.to(p.dtype).clone();w[:,:,count:]=1
    disp=d['observations']-fine.rest
    numer=torch.einsum('nc,btnk->btck',p,w[...,None]*disp);denom=torch.einsum('nc,btn->btc',p,w).clamp_min(1)
    oracle=fine.rest+p@(numer/denom[...,None]);err=(oracle-d['observations']).square().sum(-1)
    oracle_rms=float(torch.sqrt(err[:,tf+1:,:count][mask[:,tf+1:,:count]].mean())*1000)
    info=dict(strategy=name,seed=seed,level=level,nodes=n,tangent_relative_error=tangent_error,offdiag_best_scalar_relative_error=(irreducible/max(total,1e-30))**.5,elastic_force_relative_error=relative(centroid_f,exact_f),dashpot_force_relative_error=relative(centroid_d,exact_d),oracle_future_rms_mm=oracle_rms)
    for mode,s,force in [('centroid',coarse,None),('bundle',coarse,bundle),('tangent_matched',matched,None)]:
     t=time.perf_counter();y=rollout(s,pool_state(d['x0'],fine.mass,p),pool_state(d['v0'],fine.mass,p),pool_state(d['controls'],fine.mass,p),d['dt'],32,floor=0.,force_fn=force)
     pred=lift_state(y,fine,p,coarse.rest);finite=bool(torch.isfinite(pred).all())
     row=info|dict(mode=mode,finite=finite,rollout_s=time.perf_counter()-t)
     if finite:
      row['fine_reference_future_rms_mm']=float(torch.sqrt((pred[:,tf+1:,:count]-reference[:,tf+1:,:count]).square().sum(-1).mean())*1000)
      row['observation_future']=metrics(pred[:,tf+1:],d['observations'][:,tf+1:],mask[:,tf+1:],count)
      row['min_z_m']=float(pred[:,:,:count,2].min())
     rows.append(row)
    print(json.dumps(rows[-3:]),flush=True)
(root/'results/round3'/('axial_audit.json' if args.group=='round2_aligned' else 'axial_audit_new_maps.json')).write_text(json.dumps(dict(protocol=f'common initial fine system, frozen {args.group} maps, 32 substeps; internal-force exactness only',elapsed_s=time.perf_counter()-started,rows=rows),indent=2))
