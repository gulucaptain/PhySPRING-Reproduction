import sys,json,argparse
from pathlib import Path
import numpy as np
import torch
from inspect_data import load
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.model import fps
from physpring.physics import System,laplacian
from physpring.data import save_scene
p=argparse.ArgumentParser();p.add_argument('--nodes',type=int,default=64);p.add_argument('--case',default='double_lift_cloth_1');a=p.parse_args()
torch.set_num_threads(1)
folder=root/'data/different_types'/a.case;raw=load(folder/'final_data.pkl');split=json.loads((folder/'split.json').read_text())
obj=torch.tensor(raw['object_points'],dtype=torch.float32);ctrl=torch.tensor(raw['controller_points'],dtype=torch.float32)
# PhysTwin reverse_z=true -> transform to our z-up convention, also applies to controllers.
obj[:,:,2]*=-1;ctrl[:,:,2]*=-1
visible=torch.tensor(raw['object_visibilities']);valid=torch.tensor(raw['object_motions_valid'])
initial_valid=torch.where(visible[0]&valid[0]&torch.isfinite(obj[0]).all(-1))[0]
ids=initial_valid[fps(obj[0,initial_valid],a.nodes)]
ctrl_ids=fps(ctrl[0],4)
obs=torch.cat([obj[:,ids],ctrl[:,ctrl_ids]],1)[None]
rest=obs[0,0];n=len(rest);fixed=torch.arange(n)>=a.nodes
mass=torch.full((n,),1./a.nodes)
dist=torch.cdist(rest,rest);dist.fill_diagonal_(float('inf'))
adj=torch.zeros((n,n));neighbors=dist.topk(6,largest=False).indices;adj.scatter_(1,neighbors,1);adj=torch.maximum(adj,adj.T);adj[fixed[:,None]&fixed[None,:]]=0
initial=System(rest,mass,adj*18,laplacian(adj*.03)+torch.diag_embed(mass)*.1,fixed,torch.tensor([200.,2.,.2]))
mask=torch.cat([visible[:,ids]&valid[:,ids],torch.zeros((len(obj),4),dtype=torch.bool)],-1)[None]
controls=rest[None,None].expand_as(obs).clone();controls[:,:,a.nodes:]=ctrl[:,ctrl_ids][None]
d=dict(initial=initial,observations=obs,observation_mask=mask,controls=controls,x0=rest[None].clone(),v0=torch.zeros_like(rest)[None],dt=667*5e-5,substeps=667,kind='real_cloth_subset',seed=0,train_frames=split['train'][1]-1,floor=0.,source_case=a.case,object_ids=ids,controller_ids=ctrl_ids,object_count=a.nodes)
out=root/'data'/f'{a.case}_{a.nodes}.pt';save_scene(d,out)
meta=dict(case=a.case,original_object_nodes=len(obj[0]),selected_object_nodes=a.nodes,controller_nodes=4,frames=len(obj),official_split=split,initial_only_sampling=True,axis_transform='z -> -z',mass_assumption='object total mass 1kg, controller mass same per-node',graph_assumption='symmetric 6-NN',not_paper_scale=True)
(root/'configs'/f'{a.case}_{a.nodes}.json').write_text(json.dumps(meta,indent=2));print(out)
