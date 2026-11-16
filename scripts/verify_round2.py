"""Checkpoint replay and final supervised partition audit."""
import json,sys,math
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.evaluate_v2 import restore
from physpring.experiment_v2 import predict
torch.set_num_threads(1)
verification_path=root/'results/round2/verification.json'
rows=json.loads(verification_path.read_text()) if verification_path.exists() else []
checked={r['name'] for r in rows}
for folder in sorted([*(root/'results/round2').iterdir(),*(root/'results/round2_aligned').iterdir()]):
 if str(folder.relative_to(root/'results')) in checked:continue
 if not (folder/'summary.json').exists() or folder.name=='pilot':continue
 m,d,cfg=restore(folder)
 trace=[json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
 assert len(trace)==cfg['epochs']
 assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace)
 base=d['initial'];saved=torch.load(folder/'partitions.pt',weights_only=True);expected=np.load(folder/'trajectories.npz')['prediction'];levels=[]
 with torch.no_grad():
  systems,maps,_=m(base)
  for l,(s,p) in enumerate(zip(systems,maps)):
   actual=predict(s,base,p,d,cfg['actual_frames'],cfg['substeps'])
   delta=float((actual-torch.from_numpy(expected[l])).abs().max())
   same=bool(torch.equal(p,saved['maps'][l]));clusters=p.argmax(1);diameters=[];disconnected=0
   for k in range(p.shape[1]):
    indices=torch.where(clusters==k)[0].tolist();sub=base.rest[indices];diameters.append(float(torch.cdist(sub,sub).max()))
    visited={indices[0]};front=[indices[0]]
    while front:
     u=front.pop()
     for v in indices:
      if v not in visited and base.stiffness[u,v]>0:visited.add(v);front.append(v)
    disconnected+=len(visited)!=len(indices)
   levels.append(dict(level=l,checkpoint_position_max_abs_m=delta,equals_last_supervised_map=same,mass_error=float((s.mass.sum()-base.mass.sum()).abs()),disconnected_clusters=disconnected,clusters=p.shape[1],max_cluster_diameter_m=max(diameters)))
   assert np.isfinite(delta)
   if cfg['device']=='cpu':assert delta<1e-6
   if cfg['schedule']=='deterministic_tail':assert same
 rows.append(dict(name=str(folder.relative_to(root/'results')),trained_device=cfg['device'],levels=levels));print(folder.name,levels[-1],flush=True)
(root/'results/round2/verification.json').write_text(json.dumps(rows,indent=2))
