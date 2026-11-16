import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path);p.add_argument('--workers',type=int,default=6);p.add_argument('--aligned',action='store_true');p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);a=p.parse_args()
variants={
 'legacy':dict(physics='legacy',loss='legacy',schedule='legacy',heads='multiplicative',legacy_model=True),
 'loss_only':dict(physics='legacy',loss='reference',schedule='legacy',heads='multiplicative',legacy_model=True),
 'physics_only':dict(physics='reference',loss='legacy',schedule='legacy',heads='multiplicative',legacy_model=True),
 'schedule_only':dict(physics='legacy',loss='legacy',schedule='deterministic_tail',heads='multiplicative'),
 'full_global':dict(physics='reference',loss='reference',schedule='deterministic_tail',heads='additive'),
 'full_local':dict(physics='reference',loss='reference',schedule='deterministic_tail',heads='additive',merge='local'),
 'full_geometry':dict(physics='reference',loss='reference',schedule='deterministic_tail',heads='additive',strategy='geometry'),
 'full_random':dict(physics='reference',loss='reference',schedule='deterministic_tail',heads='additive',strategy='random'),
}
plan=[dict(name=n,seed=s,options=o) for n,o in variants.items() for s in a.seeds if not a.aligned or o['loss']=='reference']
output_root=(a.output_root or root/'results'/('round2_aligned' if a.aligned else 'round2')).resolve();output_root.mkdir(parents=True,exist_ok=True)
log_root=root/'logs'/output_root.name;log_root.mkdir(parents=True,exist_ok=True)
(output_root/'plan.json').write_text(json.dumps(dict(workers=a.workers,epochs=80,substeps=16,hidden=128,plan=plan),indent=2))
def run(job):
 name=f"{job['name']}_{job['seed']}";out=output_root/name
 if (out/'summary.json').exists():return name,'cached'
 if (out/'failure.json').exists():return name,'cached failure (retained)'
 if out.exists() and any(out.iterdir()):return name,'incomplete output retained; use a new output directory'
 cmd=[sys.executable,'-m','physpring.experiment_v2','--output',str(out),'--data',str(root/'data'/('double_lift_cloth_1_64_round2_aligned.pt' if a.aligned else 'double_lift_cloth_1_64_round2.pt')),'--epochs','80','--substeps','16','--hidden','128','--seed',str(job['seed'])]
 for k,v in job['options'].items():
  cmd+=['--'+k.replace('_','-')]+([] if v is True else [str(v)])
 with (log_root/f'{name}.log').open('w') as f:
  r=subprocess.run(cmd,env=os.environ|{'PYTHONPATH':str(root/'src'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT)
 return name,r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
 futures=[ex.submit(run,j) for j in plan]
 for f in concurrent.futures.as_completed(futures):print(f.result(),flush=True)
