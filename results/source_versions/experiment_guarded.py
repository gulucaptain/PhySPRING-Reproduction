import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from .data import make_scene,save_scene,load_scene
from .physics import pool_state,lift_state,rollout
from .model import PhySpring


def metrics(pred,target,mask=None,object_count=None):
    if object_count is not None:
        pred=pred[:,:,:object_count];target=target[:,:,:object_count]
        if mask is not None:mask=mask[:,:,:object_count]
    error=torch.linalg.vector_norm(pred-target,dim=-1)
    dist=torch.sqrt(((pred[:,:,:,None,:]-target[:,:,None,:,:])**2).sum(-1)+1e-12)
    if mask is None:
        cd=.5*(dist.min(-1).values.mean()+dist.min(-2).values.mean())
    else:
        dist=dist.masked_fill(~mask[:,:,None,:],1e3)
        frame_valid=mask.any(-1)
        cd=.5*((dist.min(-1).values.mean(-1)*frame_valid).sum()/frame_valid.sum().clamp_min(1)+(dist.min(-2).values*mask).sum()/mask.sum().clamp_min(1))
        error=error[mask]
    return dict(track_m=float(error.mean()) if error.numel() else None,cd_m=float(cd) if error.numel() else None,max_error_m=float(error.max()) if error.numel() else None,valid_points=int(error.numel()),valid_frames=int(mask.any(-1).sum()) if mask is not None else pred.shape[0]*pred.shape[1])


def predict(s,base,p,d,frames,substeps):
    controls=pool_state(d['controls'][:,:frames+1],base.mass,p)
    x=pool_state(d['x0'],base.mass,p);v=pool_state(d['v0'],base.mass,p)
    pred=rollout(s,x,v,controls,d['dt'],substeps,mode='axial',floor=d.get('floor',0.))
    return lift_state(pred,base,p,s.rest)


def sync(device):
    if device=='mps':torch.mps.synchronize()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--strategy',default='learned',choices=['learned','random','geometry'])
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--size',type=int,default=5);p.add_argument('--frames',type=int,default=24)
    p.add_argument('--batch',type=int,default=3);p.add_argument('--substeps',type=int,default=8)
    p.add_argument('--hidden',type=int,default=128);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--kind',default='cloth');p.add_argument('--data')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise SystemExit('Result exists; choose a fresh output directory.')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=args.device
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Apple GPU unavailable in this execution context')
    d=load_scene(args.data) if args.data else make_scene(args.kind,args.size,args.frames,args.batch,substeps=max(32,args.substeps*2))
    save_scene(d,out/'data.pt')
    for k,v in d.items():
        if isinstance(v,torch.Tensor):d[k]=v.to(device)
    base=d['initial'].to(device);d['initial']=base
    train_frames=d.get('train_frames',(d['observations'].shape[1]-1)//2);frames=d['observations'].shape[1]-1
    # Dataset construction fixes only data randomness; reset model/assignment seed afterwards.
    torch.manual_seed(args.seed)
    model=PhySpring(hidden=args.hidden,strategy=args.strategy,seed=args.seed).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    config=vars(args)|dict(train_frames=train_frames,dt=d['dt'],data_seed=d.get('seed'),train_split='data-specified or first half; future withheld',torch=torch.__version__,python=platform.python_version(),machine=platform.machine(),actual_nodes=len(base.mass),actual_batch=d['x0'].shape[0],actual_frames=frames,source_kind=d['kind'])
    (out/'config.json').write_text(json.dumps(config,indent=2))
    trace=[];start=time.perf_counter();peak=0
    for epoch in range(args.epochs):
        stage=min(3,epoch*4//args.epochs)
        model.temperature=max(.2,1.-.8*epoch/max(1,args.epochs-1))
        if epoch>0 and stage>min(3,(epoch-1)*4//args.epochs) and stage>1:model.commit(base,stage-2)
        torch.save({'model':model.state_dict(),'frozen':{k:v.cpu() for k,v in model.frozen.items()},'optimizer':optimizer.state_dict(),'config':config,'epoch':epoch,'rng':torch.get_rng_state()},out/'last_state.pt')
        optimizer.zero_grad(set_to_none=True)
        systems,maps,_=model(base,stochastic=True);loss=0
        for level in range(stage+1):
            pred=predict(systems[level],base,maps[level],d,train_frames,args.substeps)
            target=d['observations'][:,:train_frames+1]
            count=d.get('object_count',pred.shape[2]);pp=pred[:,1:,:count];tt=target[:,1:,:count]
            mask=d.get('observation_mask',torch.ones_like(d['observations'][...,0],dtype=torch.bool))[:,1:train_frames+1,:count]
            tracking=((pp-tt).square().sum(-1)*mask).sum()/(3*mask.sum().clamp_min(1))
            dd=(pp[:,:,:,None,:]-tt[:,:,None,:,:]).square().sum(-1)
            dd=dd.masked_fill(~mask[:,:,None,:],1e3)
            frame_valid=mask.any(-1)
            geometry=.5*((dd.min(-1).values.mean(-1)*frame_valid).sum()/frame_valid.sum().clamp_min(1)+(dd.min(-2).values*mask).sum()/mask.sum().clamp_min(1))
            loss=loss+tracking+geometry
        if not torch.isfinite(loss):
            (out/'failure.json').write_text(json.dumps(dict(epoch=epoch,reason='nonfinite loss')))
            raise RuntimeError(f'Nonfinite loss at epoch {epoch}; pre-update state saved')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        if not torch.isfinite(norm):
            (out/'failure.json').write_text(json.dumps(dict(epoch=epoch,reason='nonfinite gradient',loss=float(loss.detach()))))
            raise RuntimeError(f'Nonfinite gradient at epoch {epoch}; pre-update state saved')
        optimizer.step()
        if (epoch+1)%20==0:
            for g in optimizer.param_groups:g['lr']*=.9
        sync(device)
        if device=='mps':peak=max(peak,torch.mps.driver_allocated_memory())
        row=dict(epoch=epoch,stage=stage,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.perf_counter()-start)
        trace.append(row)
        with (out/'trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if epoch%10==0 or epoch==args.epochs-1:print(json.dumps(row),flush=True)
    for level in range(3):
        if level not in model.frozen:model.commit(base,level)
    torch.save({'model':model.state_dict(),'frozen':{k:v.cpu() for k,v in model.frozen.items()},'optimizer':optimizer.state_dict(),'config':config},out/'checkpoint.pt')
    training_s=time.perf_counter()-start;all_preds=[];results=[]
    with torch.no_grad():
        systems,maps,_=model(base)
        for level,(s,pmap) in enumerate(zip(systems,maps)):
            pred=predict(s,base,pmap,d,frames,args.substeps)
            if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite evaluation')
            all_preds.append(pred.cpu().numpy())
            sync(device);t=time.perf_counter()
            for _ in range(3):predict(s,base,pmap,d,frames,args.substeps)
            sync(device);secs=(time.perf_counter()-t)/3
            results.append(dict(level=level,nodes=len(s.mass),train=metrics(pred[:,1:train_frames+1],d['observations'][:,1:train_frames+1],d['observation_mask'][:,1:train_frames+1] if 'observation_mask' in d else None,d.get('object_count')),future=metrics(pred[:,train_frames+1:],d['observations'][:,train_frames+1:],d['observation_mask'][:,train_frames+1:] if 'observation_mask' in d else None,d.get('object_count')),rollout_s=secs,aggregate_fps=d['x0'].shape[0]*frames/secs))
    with torch.no_grad():
        identity=torch.eye(len(base.mass),device=device)
        initial_pred=predict(base,base,identity,d,frames,args.substeps)
        static_pred=base.rest[None,None].expand_as(d['observations'])
        baselines={name:metrics(pp[:,train_frames+1:],d['observations'][:,train_frames+1:],d['observation_mask'][:,train_frames+1:] if 'observation_mask' in d else None,d.get('object_count')) for name,pp in [('initial_untrained',initial_pred),('static',static_pred)]}
    summary=dict(baselines=baselines,status='real-data subset; not full paper benchmark' if args.data else 'procedural diagnostic; not paper benchmark',config=config,training_s=training_s,mps_driver_peak_bytes=peak,levels=results)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'trajectories.npz',prediction=np.stack(all_preds),observation=d['observations'].cpu().numpy(),rest=base.rest.cpu().numpy())
    hashes={str(f.relative_to(Path(__file__).parents[2])):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
