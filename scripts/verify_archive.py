"""Verify published scalar evidence without excluded datasets or checkpoints."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--skip-hashes',action='store_true');args=p.parse_args()
manifest=json.loads((root/'sources/published_artifacts.json').read_text())
if not args.skip_hashes:
 for entry in manifest['files']:
  path=root/entry['path'];assert path.is_file(),path
  assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],f'Published artifact changed: {path}'
completed=[]
for path in sorted((root/'results').glob('**/summary.json')):
 d=json.loads(path.read_text())
 if not isinstance(d,dict) or 'config' not in d or 'epochs' not in d['config']:continue
 trace_path=path.parent/'trace.jsonl'
 assert trace_path.exists(),path
 trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
 assert len(trace)==d['config']['epochs'],(path,'incomplete trace')
 assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace),path
 assert len(d['levels'])==4
 assert all(math.isfinite(l['future']['track_m']) for l in d['levels']),path
 completed.append(str(path.relative_to(root)))
assert len(completed)==88,f'Expected 88 archived completed runs, got {len(completed)}'
failure=json.loads((root/'results/round2/physics_only_2/failure.json').read_text())
assert failure['epoch']==67 and failure['reason']=='nonfinite loss'
rows=list(csv.DictReader((root/'results/round3/per_run.csv').open()))
assert len(rows)==72
for row in rows:
 s=json.loads((root/row['path']).read_text());l=s['levels'][int(row['level'])]
 assert math.isclose(float(row['future_track_mm']),l['future']['track_m']*1000,abs_tol=1e-10)
stats=json.loads((root/'results/round3/aggregate.json').read_text())['training']
for stat in stats:
 for level,key in [(0,'l0'),(3,'l3')]:
  values=[float(r['future_track_mm']) for r in rows if r['variant']==stat['variant'] and int(r['level'])==level]
  assert len(values)==3
  assert np.allclose([np.mean(values),np.std(values,ddof=1)],stat[key],rtol=0,atol=1e-10)
for file,count in [('axial_audit.json',108),('axial_audit_new_maps.json',36)]:
 records=json.loads((root/'results/round3'/file).read_text())['rows']
 assert len(records)==count and all(r['finite'] for r in records)
print(f'Verified {len(completed)} completed-run records, preserved failure, 72 final CSV rows, 144 constitutive diagnostics; '+('hashes skipped.' if args.skip_hashes else f"{len(manifest['files'])} published artifact hashes."))
print('Excluded checkpoints/data were not replayed; paper equivalence is not asserted.')
