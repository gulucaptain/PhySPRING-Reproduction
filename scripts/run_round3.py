import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,default=root/'results/round3');parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
output_root=args.output_root.resolve();output_root.mkdir(parents=True,exist_ok=True)
log_root=root/'logs'/output_root.name;log_root.mkdir(parents=True,exist_ok=True)
variants={
 'full_global':dict(merge='global'), 'full_local':dict(merge='local'),
 'full_geometry':dict(strategy='geometry'), 'full_random':dict(strategy='random'),
 'active_global':dict(merge='global',supervision='active'),
 'active_local':dict(merge='local',supervision='active'),
}
plan=[dict(name=n,seed=s,options=o) for n,o in variants.items() for s in range(3)]
(output_root/'plan.json').write_text(json.dumps(dict(workers=args.workers,epochs=80,substeps=16,hidden=128,projection='full',plan=plan),indent=2))
def run(j):
 name=f"{j['name']}_{j['seed']}";out=output_root/name
 if out.exists() and any(out.iterdir()):return name,'existing evidence retained'
 cmd=[sys.executable,'-m','physpring.experiment_v3','--output',str(out),'--data',str(root/'data/double_lift_cloth_1_64_round2_aligned.pt'),'--epochs','80','--substeps','16','--hidden','128','--seed',str(j['seed'])]
 for k,v in j['options'].items():cmd+=['--'+k,str(v)]
 with (log_root/f'{name}.log').open('w') as f:r=subprocess.run(cmd,env=os.environ|{'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT)
 return name,r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
 for f in concurrent.futures.as_completed([ex.submit(run,j) for j in plan]):print(f.result(),flush=True)
