"""Inspect public ZIP central directory without downloading the 8 GB archive."""
import io,json,urllib.request,zipfile,time
from pathlib import Path
class Remote(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.cache={}
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  if n==0:return b''
  start=self.pos;end=min(self.size,start+n)-1
  req=urllib.request.Request(self.url+f'?download=true&range_start={start}',headers={'Range':f'bytes={start}-{end}'})
  with urllib.request.urlopen(req,timeout=60) as r:
   if r.status!=206:raise RuntimeError(f'Range request returned {r.status}; refuse full archive download')
   data=r.read(n)
  self.pos+=len(data);return data
if __name__=='__main__':
 root=Path(__file__).resolve().parents[1]
 url='https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip'
 remote=Remote(url,8075884740)
 with zipfile.ZipFile(remote) as z:
  info=[dict(path=i.filename,size=i.file_size,compressed=i.compress_size) for i in z.infolist()]
  (root/'sources/phystwin_archive_index.json').write_text(json.dumps(info,indent=2))
  print('files',len(info));print('\n'.join(str(x) for x in info[:40]))
