"""Replay the saved pre-update topology; do not replace failed trial results."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from physpring.model import PhySpring as Legacy
from physpring.model_v2 import PhySpring
from physpring.data import load_scene
from physpring.experiment_v2 import predict
from physpring.reduction_audit import axial_tangent
p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();folder=Path(a.folder)
torch.set_num_threads(1);ck=torch.load(folder/'last_state.pt',weights_only=True);cfg=ck['config'];d=load_scene(folder/'data.pt');base=d['initial']
kw=dict(hidden=cfg['hidden'],strategy=cfg['strategy'],seed=cfg['seed']);m=Legacy(**kw) if cfg['legacy_model'] else PhySpring(**kw,merge_rule=cfg['merge'],head_rule=cfg['heads'])
m.load_state_dict(ck['model']);m.frozen=ck['frozen'];epoch=ck['epoch'];stage=min(3,epoch*4//cfg['epochs']);m.temperature=max(.2,1-.8*epoch/(cfg['epochs']-1))
if not cfg['legacy_model']:m.active_assignment=stage-1
torch.set_rng_state(ck['rng']);rows=[]
with torch.no_grad():
 systems,maps,_=m(base,True)
 for l in range(stage+1):
  s=systems[l];free=(~s.fixed).repeat_interleave(3);mass=s.mass.repeat_interleave(3)[free].double();hess=axial_tangent(s).double()[free][:,free];freq=float(torch.linalg.eigvalsh(hess/mass.sqrt()[:,None]/mass.sqrt()[None,:]).max().clamp_min(0).sqrt())
  for steps in [16,32,64,667]:
   start=time.perf_counter();y=predict(s,base,maps[l],d,cfg['train_frames'],steps)
   rows.append(dict(epoch=epoch,level=l,substeps=steps,finite=bool(torch.isfinite(y).all()),max_abs_position_m=float(y.abs().max()) if torch.isfinite(y).all() else None,linearized_h_omega=d['dt']/steps*freq,seconds=time.perf_counter()-start));print(json.dumps(rows[-1]),flush=True)
(folder/'failure_replay.json').write_text(json.dumps(rows,indent=2))
