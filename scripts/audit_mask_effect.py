"""Paired comparison of pre-alignment and corrected-mask trials; no cherry-picking."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];rows=[]
for folder in sorted((root/'results/round2_aligned').iterdir()):
 new=folder/'summary.json';old=root/'results/round2'/folder.name/'summary.json'
 if not new.exists() or not old.exists():continue
 a=json.loads(old.read_text());b=json.loads(new.read_text())
 for l in range(4):
  av=a['levels'][l]['future']['track_m']*1000;bv=b['levels'][l]['future']['track_m']*1000
  rows.append(dict(name=folder.name,level=l,before_mm=av,aligned_mm=bv,change_mm=bv-av))
(root/'results/round2/mask_alignment_effect.json').write_text(json.dumps(rows,indent=2))
print(f'{len(rows)} paired level comparisons saved')
