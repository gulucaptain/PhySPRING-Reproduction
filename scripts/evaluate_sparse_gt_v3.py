"""Independent annotated-track check; interpolation differs from original evaluator."""
import json
from pathlib import Path
import numpy as np
from inspect_data import load
root=Path(__file__).resolve().parents[1]
gt=load(root/'data/different_types/double_lift_cloth_1/gt_track_3d.pkl').copy();gt[:,:,2]*=-1
initial=np.isfinite(gt[0]).all(-1);gt=gt[:,initial]
rows=[]
paths=[p for group in ['round3'] for p in (root/'results'/group).glob('*/trajectories.npz') if p.parent.name!='pilot' and (p.parent/'config.json').exists() and json.loads((p.parent/'config.json').read_text()).get('data')]
for path in paths:
 if not path.exists():continue
 data=np.load(path);rest=data['rest'][:64];pred=data['prediction'][:,:,:,:64]
 # Fixed 3-NN inverse-distance interpolation, based exclusively on time zero.
 distance=np.linalg.norm(gt[0,:,None]-rest[None],axis=-1);ids=np.argsort(distance,axis=-1)[:,:3]
 d=np.take_along_axis(distance,ids,axis=-1);weights=1/np.maximum(d,1e-6);weights/=weights.sum(-1,keepdims=True)
 displacements=pred[:,0]-rest[None,None]
 sampled=gt[0][None,None]+(displacements[:,:,ids]*weights[None,None,:,:,None]).sum(-2)
 mask=np.isfinite(gt).all(-1)
 for level in range(4):
  err=np.linalg.norm(sampled[level]-gt,axis=-1)
  future=err[81:][mask[81:]]
  rows.append(dict(run=str(path.parent.relative_to(root)),level=level,annotated_tracks=int(initial.sum()),future_valid_points=int(mask[81:].sum()),future_error_mm=float(future.mean()*1000)))
(root/'results/round3/sparse_gt.json').write_text(json.dumps(dict(protocol='3-NN inverse-distance displacement interpolation from first-frame positions; no ground-truth future used for prediction',results=rows),indent=2));print(json.dumps(rows[:4],indent=2))
