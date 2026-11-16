import argparse,json,time
from pathlib import Path
import torch
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
from .data import load_scene
from .model import PhySpring
from .experiment import predict,metrics


def restore(folder,device='cpu'):
    folder=Path(folder);ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    cfg=ck['config'];model=PhySpring(hidden=cfg['hidden'],strategy=cfg['strategy'],seed=cfg['seed']).to(device)
    model.load_state_dict(ck['model']);model.frozen={k:v.to(device) for k,v in ck['frozen'].items()};model.eval()
    d=load_scene(folder/'data.pt',device);return model,d,cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--substeps',type=int,nargs='+',default=[8,16,32,64,667]);a=p.parse_args()
    torch.set_num_threads(1);folder=Path(a.folder);model,d,cfg=restore(folder);base=d['initial'];rows=[]
    with torch.no_grad():
        systems,maps,_=model(base)
        for level,(s,pmap) in enumerate(zip(systems,maps)):
            previous=None
            for steps in a.substeps:
                start=time.perf_counter();pred=predict(s,base,pmap,d,cfg['actual_frames'],steps)
                train=cfg['train_frames'];mask=d.get('observation_mask')
                row=dict(level=level,substeps=steps,elapsed_s=time.perf_counter()-start,finite=bool(torch.isfinite(pred).all()),future=metrics(pred[:,train+1:],d['observations'][:,train+1:],mask[:,train+1:] if mask is not None else None,d.get('object_count')))
                if previous is not None:row['max_position_change_from_previous_m']=float((pred-previous).abs().max())
                previous=pred;rows.append(row);print(json.dumps(row),flush=True)
    (folder/'substep_sensitivity.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
