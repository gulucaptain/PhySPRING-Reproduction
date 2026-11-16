import hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1];checked=[]
paths=[p for group in ['suite_final','real_final'] for p in (root/'results'/group).glob('*/summary.json')]
paths += [root/'results'/g/'summary.json' for g in ['real_repaired_0','gpu_large','real_step64_sensitivity'] if (root/'results'/g/'summary.json').exists()]
for p in paths:
  d=json.loads(p.read_text());trace=[json.loads(x) for x in (p.parent/'trace.jsonl').read_text().splitlines()]
  assert len(trace)==80,(p,'epochs')
  assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace),(p,'finite')
  assert len(d['levels'])==4
  assert all(math.isfinite(l['future']['track_m']) for l in d['levels'])
  assert (p.parent/'checkpoint.pt').exists() and (p.parent/'trajectories.npz').exists()
  checked.append(str(p.relative_to(root)))
(root/'results/verification.json').write_text(json.dumps(dict(completed_checked=len(checked),runs=checked,not_checked=['historical pilots','paper equivalence']),indent=2))
print(f'{len(checked)} completed runs verified; historical failures are not counted as completed.')
