import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'results/report';out.mkdir(exist_ok=True)
rows=[]
for group in ['suite_final','real_final']:
 for path in sorted((root/'results'/group).glob('*/summary.json')):
  r=json.loads(path.read_text());c=r['config']
  if group=='real_final' and c['strategy']=='learned' and c['seed']==0 and (root/'results/real_repaired_0/summary.json').exists():continue
  for l in r['levels']:rows.append(dict(group=group,kind=c['source_kind'],strategy=c['strategy'],seed=c['seed'],level=l['level'],nodes=l['nodes'],track_mm=l['future']['track_m']*1000,cd_mm=l['future']['cd_m']*1000,fps=l['aggregate_fps'],speedup=l['aggregate_fps']/r['levels'][0]['aggregate_fps'],source=str(path.relative_to(root))))
repair=root/'results/real_repaired_0/summary.json'
if repair.exists():
 r=json.loads(repair.read_text());c=r['config']
 for l in r['levels']:rows.append(dict(group='real_final',kind=c['source_kind'],strategy=c['strategy'],seed=c['seed'],level=l['level'],nodes=l['nodes'],track_mm=l['future']['track_m']*1000,cd_mm=l['future']['cd_m']*1000,fps=l['aggregate_fps'],speedup=l['aggregate_fps']/r['levels'][0]['aggregate_fps'],source=str(repair.relative_to(root))))
with (out/'per_run.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for kind in sorted({r['kind'] for r in rows}):
 for strategy in ['learned','geometry','random']:
  for level in range(4):
   rs=[r for r in rows if r['kind']==kind and r['strategy']==strategy and r['level']==level]
   if not rs:continue
   v=np.array([r['track_mm'] for r in rs]);summary.append(dict(kind=kind,strategy=strategy,level=level,n=len(rs),track_mean_mm=float(v.mean()),track_std_mm=float(v.std(ddof=1)) if len(v)>1 else 0.,speedup=float(np.mean([r['speedup'] for r in rs]))))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
for ax,kind in zip(axes,['cloth','rope','real_cloth_subset']):
 for strategy in ['learned','geometry','random']:
  rs=[r for r in summary if r['kind']==kind and r['strategy']==strategy]
  if rs:ax.errorbar([r['level'] for r in rs],[r['track_mean_mm'] for r in rs],yerr=[r['track_std_mm'] for r in rs],marker='o',label=strategy,capsize=3)
 ax.set(title=kind.replace('_',' '),xlabel='Reduction level',ylabel='Future tracking error (mm)',xticks=range(4));ax.grid(alpha=.2);ax.legend()
fig.suptitle('Independent implementation: local diagnostics, NOT paper benchmark reproduction',fontsize=12)
fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
# Compare actual tracked geometry and predictions over time for the first real seed.
path=(root/'results/real_repaired_0/trajectories.npz') if repair.exists() else root/'results/real_final/learned_1/trajectories.npz'
if path.exists():
 data=np.load(path);obs=data['observation'][0];pred=data['prediction'][:,0]
 fig,axes=plt.subplots(2,3,figsize=(11,7),subplot_kw={'projection':'3d'},layout='constrained')
 for col,t in enumerate([0,80,115]):
  for row,level in enumerate([0,3]):
   ax=axes[row,col];gt=obs[t,:64];pp=pred[level,t,:64]
   ax.scatter(*gt.T,s=10,c='#555555',label='observed');ax.scatter(*pp.T,s=10,c='#D85B35',label=f'L{level}')
   ax.set(title=f'Frame {t}, L{level}',xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');ax.view_init(20,-60);ax.legend(fontsize=8)
 fig.suptitle('PhysTwin cloth: 64 selected points; raw tracks shown (including invalid tracks)',fontsize=11)
 fig.savefig(out/'real_trajectories.png',dpi=160);plt.close(fig)
lines=['# 本地复现结果','', '以下均为独立实现的诊断实验，不能视为原论文图表复现。均值±样本标准差；错误越低越好。','', '| 数据 | 方法 | 层 | 完成种子数 | 未来追踪误差 mm | 相对本方法L0吞吐 |','|---|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['kind']} | {r['strategy']} | {r['level']} | {r['n']} | {r['track_mean_mm']:.3f} ± {r['track_std_mm']:.3f} | {r['speedup']:.2f}× |")
lines+=['','原始16子步真实数据 learned/seed0 曾失败，见 `results/real_failure_audit/`；如有修复运行，表中明确使用 `real_repaired_0`，其余原始失败日志保留。','', '计时含投影/积分/轨迹恢复，不含渲染；多进程竞争影响CPU计时。不能与论文RTX4090端到端FPS直接比较。']
(out/'results.md').write_text('\n'.join(lines)+'\n');print(out)
