import argparse
import concurrent.futures,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output-root',default='results/real_final');args=p.parse_args();destination=root/args.output_root
def run(job):
 strategy,seed=job;name=f'{strategy}_{seed}';out=destination/name
 if (out/'summary.json').exists():return name,0
 if args.output_root=='results/real_final' and strategy=='learned' and seed==0 and (root/'results/real_repaired_0/summary.json').exists():return name,0
 cmd=[sys.executable,'-m','physpring.experiment','--output',str(out),'--data',str(root/'data/double_lift_cloth_1_64.pt'),'--strategy',strategy,'--seed',str(seed),'--epochs','80','--substeps','16','--hidden','128']
 with (root/'logs'/f'real_{name}.log').open('w') as f:
  r=subprocess.run(cmd,env=os.environ|{'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT)
 return name,r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 for x in ex.map(run,[(s,i) for s in ['learned','geometry','random'] for i in [0,1,2]]):print(x,flush=True)
