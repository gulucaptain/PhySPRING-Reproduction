"""Restore round-2 checkpoints without redrawing frozen assignments."""
import argparse,json,time
from pathlib import Path
import torch
from .data import load_scene
from .model import PhySpring as LegacyPhySpring
from .model_v2 import PhySpring
from .experiment_v2 import predict,metrics

def restore(folder,device='cpu'):
    folder=Path(folder)
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    cfg=ck['config']
    kwargs=dict(hidden=cfg['hidden'],strategy=cfg['strategy'],seed=cfg['seed'])
    model=(LegacyPhySpring(**kwargs) if cfg['legacy_model'] else PhySpring(**kwargs,merge_rule=cfg['merge'],head_rule=cfg['heads'])).to(device)
    model.load_state_dict(ck['model']);model.frozen={k:v.to(device) for k,v in ck['frozen'].items()};model.eval()
    return model,load_scene(folder/'data.pt',device),cfg

def main():
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--substeps',type=int,nargs='+',default=[16,32,64,667]);p.add_argument('--levels',type=int,nargs='+',default=[0,3]);a=p.parse_args()
    torch.set_num_threads(1);folder=Path(a.folder);model,d,cfg=restore(folder);base=d['initial'];rows=[]
    with torch.no_grad():
        systems,maps,_=model(base)
        for level in a.levels:
            s,pmap=systems[level],maps[level];previous=None
            for steps in a.substeps:
                start=time.perf_counter();pred=predict(s,base,pmap,d,cfg['actual_frames'],steps)
                train=cfg['train_frames'];mask=d.get('observation_mask')
                row=dict(level=level,substeps=steps,elapsed_s=time.perf_counter()-start,finite=bool(torch.isfinite(pred).all()),future=metrics(pred[:,train+1:],d['observations'][:,train+1:],mask[:,train+1:] if mask is not None else None,d.get('object_count')))
                if previous is not None:row['max_position_change_from_previous_m']=float((pred-previous).abs().max())
                previous=pred;rows.append(row);print(json.dumps(row),flush=True)
    (folder/'substep_sensitivity.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
