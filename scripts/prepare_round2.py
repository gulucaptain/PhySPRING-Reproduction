"""Keep round1 sampling/graph exactly, add the two original supervision masks."""
import sys,json,hashlib,argparse
from pathlib import Path
import torch
from inspect_data import load
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.data import load_scene,save_scene
from physpring.masks import align_motion_mask
parser=argparse.ArgumentParser();parser.add_argument('--legacy-mask',action='store_true',help='Rebuild superseded audit data with the old same-frame motion mask');args=parser.parse_args()
d=load_scene(root/'data/double_lift_cloth_1_64.pt');raw=load(root/'data/different_types/double_lift_cloth_1/final_data.pkl')
ids=d['object_ids'];frames=d['observations'].shape[1];nc=d['controls'].shape[2]-len(ids)
for key,source in [('geometry_mask','object_visibilities'),('tracking_mask','object_motions_valid')]:
 raw_mask=torch.tensor(raw[source])[:,ids]
 finite=torch.isfinite(d['observations'][0,:,:len(ids)]).all(-1)
 mask=align_motion_mask(raw_mask,finite) if key=='tracking_mask' and not args.legacy_mask else raw_mask&finite
 d[key]=torch.cat([mask,torch.zeros((frames,nc),dtype=torch.bool)],-1)[None]
d['tracking_mask_convention']='superseded: target frame t uses motion_valid[t]' if args.legacy_mask else 'target frame t uses motion_valid[t-1]; t=0 unused'
save_scene(d,root/'data'/('double_lift_cloth_1_64_round2.pt' if args.legacy_mask else 'double_lift_cloth_1_64_round2_aligned.pt'))
print({key:int(d[key].sum()) for key in ['observation_mask','geometry_mask','tracking_mask']})
