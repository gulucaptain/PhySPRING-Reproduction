import sys,json
from pathlib import Path
import torch
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.data import load_scene
from physpring.model import PhySpring
from physpring.experiment import predict
folder=root/'results/real_failure_audit';ck=torch.load(folder/'last_state.pt',map_location='cpu',weights_only=True)
torch.set_num_threads(1);d=load_scene(folder/'data.pt');base=d['initial'];epoch=ck['epoch'];cfg=ck['config']
rows=[]
for steps in [16,32,64]:
 model=PhySpring(hidden=cfg['hidden'],seed=cfg['seed']);model.load_state_dict(ck['model']);model.frozen=ck['frozen'];model.temperature=max(.2,1-.8*epoch/79)
 torch.set_rng_state(ck['rng']);systems,maps,_=model(base,True);loss=0;tf=cfg['train_frames'];count=d['object_count'];mask=d['observation_mask'][:,1:tf+1,:count]
 for level in range(min(3,epoch*4//80)+1):
  pred=predict(systems[level],base,maps[level],d,tf,steps);pp=pred[:,1:,:count];tt=d['observations'][:,1:tf+1,:count]
  tracking=((pp-tt).square().sum(-1)*mask).sum()/(3*mask.sum().clamp_min(1))
  dd=(pp[:,:,:,None,:]-tt[:,:,None,:,:]).square().sum(-1);dd=dd.masked_fill(~mask[:,:,None,:],1e3);fv=mask.any(-1)
  loss=loss+tracking+.5*((dd.min(-1).values.mean(-1)*fv).sum()/fv.sum().clamp_min(1)+(dd.min(-2).values*mask).sum()/mask.sum().clamp_min(1))
 loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
 row=dict(epoch=epoch,substeps=steps,loss=float(loss.detach()),all_gradient_elements_finite=all(bool(torch.isfinite(g).all()) for g in grads),max_gradient=float(max(g.abs().max() for g in grads)),norm_float64=float(torch.sqrt(sum(g.double().square().sum() for g in grads))),norm_float32=float(torch.sqrt(sum(g.square().sum() for g in grads))))
 rows.append(row);print(json.dumps(row),flush=True)
(folder/'gradient_audit.json').write_text(json.dumps(rows,indent=2))
