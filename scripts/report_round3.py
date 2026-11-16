import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'results/round3';plan=json.loads((root/'configs/round3_plan.json').read_text());rows=[];stats=[]
for job in plan['plan']:
 name=f"{job['name']}_{job['seed']}";p=out/name/'summary.json'
 if not p.exists():raise RuntimeError(f'Unfinished or failed: {name}; inspect saved log')
 d=json.loads(p.read_text())
 for l in [0,1,2,3]:
  rows.append(dict(variant=job['name'],seed=job['seed'],level=l,future_track_mm=d['levels'][l]['future']['track_m']*1000,train_track_mm=d['levels'][l]['train']['track_m']*1000,training_s=d['training_s'],path=str(p.relative_to(root))))
with (out/'per_run.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Round 3 controlled results','', 'Single real cloth subset; 64 object + 4 controller nodes; seeds 0/1/2; 80 updates; 16 substeps; future common-mask Euclidean observation error. Mean ± sample SD. Not the complete paper benchmark.','', '| Variant | L0 future (mm) | L3 future (mm) |','|---|---:|---:|']
for name in dict.fromkeys(r['variant'] for r in rows):
 vals=[]
 for l in [0,3]:
  v=[r['future_track_mm'] for r in rows if r['variant']==name and r['level']==l];vals.append((float(np.mean(v)),float(np.std(v,ddof=1))))
 stats.append(dict(variant=name,l0=vals[0],l3=vals[1]));lines.append('| '+name+' | '+' | '.join(f'{m:.3f} ± {sd:.3f}' for m,sd in vals)+' |')
strategies=['full_global','full_local','full_geometry','full_random'];old=[];paired=[]
for n in strategies:
 v=[]
 for seed in range(3):
  prev=json.loads((root/'results/round2_aligned'/f'{n}_{seed}/summary.json').read_text())['levels'][3]['future']['track_m']*1000
  cur=next(r['future_track_mm'] for r in rows if r['variant']==n and r['seed']==seed and r['level']==3)
  paired.append(dict(variant=n,seed=seed,round2_mm=prev,round3_mm=cur,change_mm=cur-prev));v.append(prev)
 old.append((float(np.mean(v)),float(np.std(v,ddof=1))))
lines+=['','## Internal-force reduction diagnostics','', 'Frozen ROUND-2 maps, common initial fine-system coefficients, 32 substeps. No parameter fitting. RMS relative to the common fine rollout is distinct from Euclidean error against observations. Oracle RMS uses future data only for hindsight representation analysis.','', '| Frozen map | Tangent error | Oracle RMS (mm) | Fine-rollout RMS: centroid / bundle / matched (mm) | Observation mean: centroid / bundle / matched (mm) |','|---|---:|---:|---|---|']
audit=json.loads((out/'axial_audit.json').read_text())['rows'];force_stats=[]
for n in strategies:
 g=[r for r in audit if r['strategy']==n and r['level']==3];base=[r for r in g if r['mode']=='centroid'];f=[];obs=[]
 for mode in ['centroid','bundle','tangent_matched']:
  sub=[r for r in g if r['mode']==mode];f.append(float(np.mean([r['fine_reference_future_rms_mm'] for r in sub])));obs.append(float(np.mean([r['observation_future']['track_m']*1000 for r in sub])))
 tangent=float(np.mean([r['tangent_relative_error'] for r in base]));oracle=float(np.mean([r['oracle_future_rms_mm'] for r in base]));force_stats.append(dict(variant=n,tangent=tangent,oracle=oracle,fine_rms=f,observation_mean=obs))
 lines.append(f'| {n} | {100*tangent:.1f}% | {oracle:.2f} | '+ ' / '.join(f'{x:.2f}' for x in f)+' | '+' / '.join(f'{x:.2f}' for x in obs)+' |')
(out/'results.md').write_text('\n'.join(lines)+'\n');(out/'aggregate.json').write_text(json.dumps(dict(training=stats,paired=paired,force=force_stats),indent=2))
fig,axes=plt.subplots(1,2,figsize=(13,4.8));x=np.arange(4);labels=['Learned global','Learned local','Geometry','Random reps']
axes[0].bar(x-.2,[r[0] for r in old],.4,yerr=[r[1] for r in old],capsize=3,label='Round 2: diagonal relaxation')
new=[next(s for s in stats if s['variant']==n)['l3'] for n in strategies]
axes[0].bar(x+.2,[r[0] for r in new],.4,yerr=[r[1] for r in new],capsize=3,label='Round 3: full mass relaxation')
axes[0].set_ylabel('Future observation error (mm)');axes[0].set_title('Trained L3: 3 seeds, mean ± sample SD');axes[0].legend(fontsize=8)
for i,(mode,label) in enumerate([('centroid','Centroid'),('bundle','Exact internal-force bundle'),('tangent_matched','Matched tangent scalar')]):
 axes[1].bar(x+(i-1)*.25,[s['fine_rms'][i] for s in force_stats],.25,label=label)
axes[1].set_ylabel('Future RMS against common fine rollout (mm)');axes[1].set_title('Fixed Round-2 maps: constitutive diagnostic');axes[1].legend(fontsize=8)
for ax in axes:ax.set_xticks(x,labels,rotation=18,ha='right');ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180)
print('\n'.join(lines))
