"""Separate hard-label changes from STE floating-point roundoff."""
import json,sys,hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from physpring.evaluate_v2 import restore
torch.set_num_threads(1);rows=[]
for group in ['round2','round2_aligned']:
 for folder in sorted((root/'results'/group).iterdir()):
  if not (folder/'summary.json').exists() or folder.name=='pilot':continue
  m,d,c=restore(folder);saved=torch.load(folder/'partitions.pt',weights_only=True)
  with torch.no_grad():_,maps,_=m(d['initial'])
  levels=[dict(level=l,changed_node_labels=int((a.argmax(-1)!=b.argmax(-1)).sum()),max_assignment_difference=float((a-b).abs().max())) for l,(a,b) in enumerate(zip(maps,saved['maps']))]
  if c['schedule']=='deterministic_tail':assert all(x['changed_node_labels']==0 and x['max_assignment_difference']==0 for x in levels)
  rows.append(dict(run=str(folder.relative_to(root)),schedule=c['schedule'],levels=levels))
  hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (folder/'source').glob('*.py')}
  (folder/'source_snapshot_hashes.json').write_text(json.dumps(hashes,indent=2))
(root/'results/round2/topology_freeze_audit.json').write_text(json.dumps(rows,indent=2));print(f'{len(rows)} topology audits saved')
