"""2x2 frozen-map/weight swaps; evaluation only, no future-data fitting."""
import json,sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.evaluate_v3 import restore
from physpring.experiment_v3 import predict,metrics

torch.set_num_threads(1);rows=[]
for name in ['full_global','full_local']:
 for seed in range(3):
  new=root/'results/round3'/f'{name}_{seed}';old=root/'results/round2_aligned'/f'{name}_{seed}'
  model,d,cfg=restore(new);base=d['initial'];cks={k:torch.load(f/'checkpoint.pt',weights_only=True) for k,f in [('old',old),('new',new)]}
  for weights in ['old','new']:
   for topology in ['old','new']:
    model.load_state_dict(cks[weights]['model']);model.frozen=cks[topology]['frozen']
    with torch.no_grad():
     systems,maps,_=model(base);pred=predict(systems[3],base,maps[3],d,cfg['actual_frames'],cfg['substeps']);tf=cfg['train_frames']
     met=metrics(pred[:,tf+1:],d['observations'][:,tf+1:],d['observation_mask'][:,tf+1:],d['object_count'])
    row=dict(variant=name,seed=seed,weights=weights,topology=topology,future_track_mm=met['track_m']*1000);rows.append(row);print(json.dumps(row),flush=True)
(root/'results/round3/map_parameter_swap.json').write_text(json.dumps(rows,indent=2))
