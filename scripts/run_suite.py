import argparse
import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output-root',default='results/suite_final');args=p.parse_args();destination=root/args.output_root
plan=[dict(kind=k,strategy=s,seed=i) for k in ['cloth','rope'] for s in ['learned','geometry','random'] for i in [0,1,2]]
(root/'configs/suite.json').write_text(json.dumps(dict(workers=6,epochs=80,size=6,frames=32,batch=3,substeps=16,hidden=128,runs=plan),indent=2))
def run(c):
 name=f"{c['kind']}_{c['strategy']}_{c['seed']}";out=destination/name
 if (out/'summary.json').exists():return name,0
 cmd=[sys.executable,'-m','physpring.experiment','--output',str(out),'--kind',c['kind'],'--strategy',c['strategy'],'--seed',str(c['seed']),'--epochs','80','--size','6','--frames','32','--batch','3','--substeps','16','--hidden','128']
 env=os.environ|{'PYTHONPATH':str(root/'src'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
 with (root/'logs'/f'final_{name}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 return name,r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 for res in ex.map(run,plan):print(res,flush=True)
