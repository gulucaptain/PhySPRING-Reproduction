import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'results/round2';plan=json.loads((root/'configs/round2_plan.json').read_text());rows=[];failures=[]
for job in plan['plan']:
 name=f"{job['name']}_{job['seed']}"
 runroot=root/'results/round2_aligned' if job['options']['loss']=='reference' else out
 path=runroot/name/'summary.json'
 if not path.exists():
  failure=runroot/name/'failure.json'
  if failure.exists():
   failures.append(dict(name=name,**json.loads(failure.read_text())));continue
  raise RuntimeError(f'Incomplete run: {name}')
 s=json.loads(path.read_text())
 for lev in s['levels']:
  rows.append(dict(variant=job['name'],seed=job['seed'],level=lev['level'],nodes=lev['nodes'],future_track_mm=lev['future']['track_m']*1000,future_cd_mm=lev['future']['cd_m']*1000,train_track_mm=lev['train']['track_m']*1000,training_s=s['training_s'],path=str(path.relative_to(root))))
with (out/'per_run.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
variants=list(dict.fromkeys(r['variant'] for r in rows));stats=[]
lines=['# Round 2: controlled real-subset audit','', '64 object points + 4 controller points, 80 training frames, 35 withheld future frames, 80 epochs, 16 integration substeps. Each variant uses seeds 0/1/2 and the same observations. Reference-loss trials use corrected transition masks from round2_aligned; original round2 reference-loss trials are superseded. Metrics below are common intersection-mask observation metrics, not official sparse annotation metrics. Error bars are sample standard deviations.','', '| Variant (completed / 3) | L0 future track (mm) | L3 future track (mm) | L3 future symmetric CD (mm) |','|---|---:|---:|---:|']
for v in variants:
 group={l:[r for r in rows if r['variant']==v and r['level']==l] for l in (0,3)}
 vals=[]
 for l,k in [(0,'future_track_mm'),(3,'future_track_mm'),(3,'future_cd_mm')]:
  a=np.array([r[k] for r in group[l]]);vals.append((float(a.mean()),float(a.std(ddof=1))))
 stats.append(dict(variant=v,completed=len(group[0]),l0=vals[0],l3=vals[1],cd=vals[2]))
 lines.append('| '+v+f' ({len(group[0])}/3)'+' | '+' | '.join(f'{m:.3f} ± {sd:.3f}' for m,sd in vals)+' |')
lines+=['', 'Failed trials (excluded from aggregates above, explicitly retained):', *[f"- {f['name']}: epoch {f['epoch']}, {f['reason']}" for f in failures], '', 'Groups with fewer than 3 completions are survivor statistics, not comparable three-seed estimates.']
(out/'failures.json').write_text(json.dumps(failures,indent=2))
(out/'results.md').write_text('\n'.join(lines)+'\n');(out/'aggregate.json').write_text(json.dumps(stats,indent=2))
fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(stats))
for i,(key,label) in enumerate([('l0','L0: 68 nodes'),('l3','L3: 20 nodes')]):
 ax.bar(x+(i-.5)*.36,[s[key][0] for s in stats],.36,yerr=[s[key][1] for s in stats],label=label,capsize=3)
ax.set_xticks(x,[s['variant']+(f" ({s['completed']}/3)" if s['completed']<3 else '') for s in stats],rotation=25,ha='right');ax.set_ylabel('Future observation tracking error (mm)');ax.set_title('PhySPRING independent reproduction — controlled corrections (mean ± sample SD)');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180)
print('\n'.join(lines))
