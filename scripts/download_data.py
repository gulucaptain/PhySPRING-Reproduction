import argparse,hashlib,json,zipfile
from pathlib import Path
from inspect_remote_data import Remote
p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['double_lift_cloth_1']);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
url='https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip'
manifest=[]
with zipfile.ZipFile(Remote(url,8075884740)) as z:
 for case in a.cases:
  for name in ['final_data.pkl','gt_track_3d.pkl','metadata.json','split.json']:
   path=f'data/different_types/{case}/{name}';dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():
    content=z.read(path);dest.write_bytes(content)
   content=dest.read_bytes();entry=dict(path=path,bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),source=url,member=path)
   manifest.append(entry);print(json.dumps(entry),flush=True)
(root/'sources/download_manifest.json').write_text(json.dumps(manifest,indent=2))
