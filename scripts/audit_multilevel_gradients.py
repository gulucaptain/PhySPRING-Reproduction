"""Descriptive shared-encoder gradient conflict at saved pre-final-update states."""
import json,sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.model_v3 import PhySpring
from physpring.data import load_scene
from physpring.experiment_v3 import predict
from physpring.losses_v2 import observation_loss

torch.set_num_threads(1);rows=[]
for name in ['full_global','full_local']:
 for seed in range(3):
  folder=root/'results/round3'/f'{name}_{seed}';ck=torch.load(folder/'last_state.pt',weights_only=True);cfg=ck['config'];d=load_scene(folder/'data.pt');base=d['initial']
  model=PhySpring(hidden=cfg['hidden'],strategy=cfg['strategy'],seed=seed,merge_rule=cfg['merge'],head_rule=cfg['heads']);model.projection_rule='full';model.load_state_dict(ck['model']);model.frozen=ck['frozen'];model.temperature=max(.2,1-.8*ck['epoch']/(cfg['epochs']-1));model.active_assignment=2
  params=[p for n,p in model.named_parameters() if n.startswith(('node_enc.','edge_enc.','enc.'))];vectors=[];losses=[]
  systems,maps,_=model(base,False);tf=cfg['train_frames'];n=d['object_count']
  for level in [0,3]:
   pred=predict(systems[level],base,maps[level],d,tf,cfg['substeps'])
   loss,_=observation_loss(pred[:,1:,:n],d['observations'][:,1:tf+1,:n],d['geometry_mask'][:,1:tf+1,:n],d['tracking_mask'][:,1:tf+1,:n])
   grads=torch.autograd.grad(loss,params,retain_graph=level==0,allow_unused=True)
   vector=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(params,grads)]).double();vectors.append(vector);losses.append(float(loss.detach()))
  a,b=vectors;row=dict(variant=name,seed=seed,epoch=ck['epoch'],l0_loss=losses[0],l3_loss=losses[1],l0_encoder_gradient_norm=float(a.norm()),l3_encoder_gradient_norm=float(b.norm()),cosine=float(a@b/(a.norm()*b.norm()).clamp_min(1e-30)));rows.append(row);print(json.dumps(row),flush=True)
(root/'results/round3/multilevel_gradient_audit.json').write_text(json.dumps(rows,indent=2))
