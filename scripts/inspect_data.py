import pickle,numpy as np,json
from pathlib import Path
class ArrayUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,('numpy._core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,('numpy.core.multiarray','scalar'):np._core.multiarray.scalar,('numpy._core.multiarray','scalar'):np._core.multiarray.scalar}
  if (module,name) not in allowed:raise pickle.UnpicklingError(f'Unsupported class {module}.{name}')
  return allowed[module,name]
def load(path):
 with open(path,'rb') as f:return ArrayUnpickler(f).load()
def describe(x,depth=0):
 if isinstance(x,dict):return {str(k):describe(v,depth+1) for k,v in x.items()}
 if isinstance(x,np.ndarray):return dict(shape=list(x.shape),dtype=str(x.dtype),min=float(x.min()) if x.size and x.dtype.kind in 'fiu' else None,max=float(x.max()) if x.size and x.dtype.kind in 'fiu' else None)
 if isinstance(x,(list,tuple)):return {'length':len(x),'first':describe(x[0],depth+1) if len(x) else None}
 return str(x)[:150]
if __name__=='__main__':
 root=Path(__file__).resolve().parents[1]
 for p in (root/'data/different_types').glob('*/*'):
  if p.suffix=='.pkl':print(p.name,json.dumps(describe(load(p)),indent=2))
  else:print(p.name,p.read_text())
